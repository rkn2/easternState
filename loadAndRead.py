#!/usr/bin/env python
# in terminal pip install ezdxf==0.6.2
import ezdxf
import os
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from matplotlib import pyplot as plt

# read the cad file
os.chdir(r"/Volumes/GoogleDrive/My Drive/Documents/Research/easternStatePenitentiary/2020_1_28_files/allOfIt")
cadFile = r"2020_01_24_DRAFT_West_Wall_mkr.dxf"
doc = ezdxf.readfile(cadFile)

# record all entities in modelspace
msp = doc.modelspace()
all_objs = [e for e in msp]
layers = sorted(set([x.dxf.layer for x in all_objs]))  # set is a list of unique items

# # get all the polylines in the designated layer
# lyr = "W-SURF-STAIN-T1"
# layer_objs = [x for x in all_objs if x.dxf.layer == lyr]
# poly = []
# for x in layer_objs:
#     if 'get_points' not in dir(x):
#         continue
#     points = np.array([y for y in x.get_points()])[:, :2]
#     poly.append(points)
# print('read %d polylines from layer %s' % (len(poly), lyr))

min_max = np.array([np.Inf * np.ones(2), -np.Inf * np.ones(2)])
for x in all_objs:
    try:
        p = np.array([y for y in x.get_points()])[:, :2]
        min_max[0] = np.min([min_max[0], np.min(p, axis=0)], axis=0)
        min_max[1] = np.max([min_max[1], np.max(p, axis=0)], axis=0)
    except AttributeError:
        pass


layerDict = {}
# get all the polylines in each layer
for lyr in layers:
    layer_objs = [x for x in all_objs if x.dxf.layer == lyr]
    polyList = []
    for x in layer_objs:
        if 'get_points' not in dir(x):
            continue
        points = np.array([y for y in x.get_points()])[:, :2]
        points = points.tolist()
        if len(points) > 2:
            poly = Polygon(points)
            if poly.is_valid:
                polyList.append(poly)
    mp = unary_union(polyList)
    if not mp.is_valid:
        raise RuntimeError
    layerDict.update({lyr: MultiPolygon(polyList)})

# pick random points
num_samples = 10000
xRand = np.random.uniform(min_max[0, 0], min_max[1, 0], num_samples)
yRand = np.random.uniform(min_max[0, 1], min_max[1, 1], num_samples)
pointList = [Point(xRand[i], yRand[i]) for i in range(num_samples)]

truthMat = np.ones([len(xRand), len(layers)], dtype=np.float) * np.Inf

# broken layers: 0, 1, 2, 3, 9
# (found this by taking sum of truthMat over axis 0, all zero means no points outside that MultiPolygon)
# suspect there are open polygons being read into shapely

for i, point in tqdm(enumerate(pointList), total=len(pointList)):
    for j, layer in enumerate(layers):
        mp = layerDict[layer]
        if len(mp) == 0:
            continue
        d = mp.distance(point)
        truthMat[i, j] = d

plt.scatter(xRand, yRand, s=1, c=truthMat[:, 4])
plt.show()
