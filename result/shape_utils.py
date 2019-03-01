from shapely.geometry import Point,Polygon
from shapely.ops import nearest_points


def get_near(edge1,edge2):

    l1 = list(edge1)
    l2 = list(edge2)
    l1.append(edge1[0])
    l2.append(edge2[0])
    l1 = [tuple(x) for x in l1]
    l2 = [tuple(x) for x in l2]
    pl1 = Polygon(l1)
    pl2 = Polygon(l2)
    print(pl1.distance(pl2))

    for o in nearest_points(pl1,pl2):
        print(o.wkt)



def simlyfy(edge1, t=1):
    l1 = list(edge1)
    l1.append(edge1[0])
    l1 = [tuple(x) for x in l1]
    pl1 = Polygon(l1)
    s = pl1.simplify(1)
    cor = list(s.exterior.coords)
    cor = [list(x) for x in cor]
    cor.pop()
    return cor

def get_area_edge(edge1):
    l1 = list(edge1)
    l1.append(edge1[0])
    l1 = [tuple(x) for x in l1]
    pl1 = Polygon(l1)
    return pl1.area

def convert_poly(edge1):
    l1 = list(edge1)
    l1.append(edge1[0])
    l1 = [tuple(x) for x in l1]
    pl1 = Polygon(l1)
    return pl1


