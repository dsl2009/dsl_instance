import torch
import torch.nn as nn
from torch.nn import functional as F
import time

def gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))

    feat = gather_feat(feat, ind)
    return feat

def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep



def _topk(scores, K=20):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def decode(
        tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr,
        K=100, kernel=1, ae_threshold=1, num_dets=1000
    ):
    batch, cat, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)

    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)

    if tl_regr is not None and br_regr is not None:
        tl_regr = tranpose_and_gather_feat(tl_regr, tl_inds)
        tl_regr = tl_regr.view(batch, K, 1, 2)
        br_regr = tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, 1, K, 2)

        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

    tl_tag = tranpose_and_gather_feat(tl_tag, tl_inds)
    tl_tag = tl_tag.view(batch, K, 1)
    br_tag = tranpose_and_gather_feat(br_tag, br_inds)
    br_tag = br_tag.view(batch, 1, K)
    dists = torch.abs(tl_tag - br_tag)

    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    scores = (tl_scores + br_scores) / 2

    # reject boxes based on classes
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tl_clses != br_clses)

    # reject boxes based on distances
    dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights
    width_inds = (br_xs < tl_xs)
    height_inds = (br_ys < tl_ys)

    scores[cls_inds] = -1
    scores[dist_inds] = -1
    scores[width_inds] = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = gather_feat(bboxes, inds)

    clses = tl_clses.contiguous().view(batch, -1, 1)
    clses = gather_feat(clses, inds).float()

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)
    tl_scores = gather_feat(tl_scores, inds).float()
    br_scores = br_scores.contiguous().view(batch, -1, 1)
    br_scores = gather_feat(br_scores, inds).float()

    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
    return detections


def neg_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)
    neg_weights = torch.pow(1 - gt[neg_inds], 4)
    print(neg_weights)
    loss = 0
    pos_pred = preds[pos_inds]
    neg_pred = preds[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


def sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return x


def ae_loss(tag0, tag1, mask):
    num = mask.sum(dim=1, keepdim=True).float()

    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    tag_mean = (tag0 + tag1) / 2
    tag0 = tag0[mask].sum()


    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push



def regr_loss(regr, gt_regr, mask):
    mask[mask>=1] =1
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)
    regr = regr[mask]
    gt_regr = gt_regr[mask]
    regr_losses = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_losses = regr_losses / (num + 1e-4)
    return regr_losses





def cluster_loss(tages, masks,dert_v=1,dert_d=1):
    nm1 = []
    nm2 = []
    distance_v = []
    vars_v = []
    for b in range(tages.shape[0]):
        tag = tages[b]
        mask = masks[b]
        means = []
        vars =[]
        for k in range(1,torch.max(mask).item()+1):
            tag_tmp= tag[mask==k]
            mean = torch.mean(tag_tmp, 0)
            for tg in range(tag_tmp.size(0)):
                var = torch.pow(torch.dist(tag_tmp[tg],mean)-dert_v, 2)
            vars.append(var)
            means.append(mean)

        distance = []
        for i in range(len(means)):
            for j in range(len(means)):
                if i!=j:
                    distance.append(torch.pow(2*dert_d-torch.dist(means[i], means[j]) , 2))

        mask[mask > 0] = 1
        nm1.append(mask.sum())
        nm2.append(len(means))
        distance_v.append(sum(distance))
        vars_v.append(sum(vars))
    var_loss = sum(vars_v)/sum(nm1)
    distance_loss = sum(distance_v)/(sum(nm2)*(sum(nm2)-1))
    return var_loss+distance_loss

def weight_be_loss(pred, target, weight):
    pred = pred.squeeze(1)
    target = target.squeeze(1)
    B, H, W = pred.size()
    total_loss =0
    for b in range(B):
        loss = F.binary_cross_entropy_with_logits(input=pred[b], target=target[b], pos_weight=torch.tensor(weight[b]))
        total_loss = total_loss+loss
    total_loss = total_loss/B
    return total_loss





def tt():
    target = torch.tensor([[1,0,1,1,1,1]]).float()
    log = torch.tensor([[1.0, 1, 1,1,1,1]]).float()
    print(F.binary_cross_entropy_with_logits(input=log, target=target, pos_weight=torch.tensor([6.0]),reduction='mean'))
    print(neg_loss(log, target))


















if __name__ == '__main__':
    tt()