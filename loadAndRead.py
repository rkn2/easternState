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

import pandas as pd
from factor_analyzer import FactorAnalyzer

from scipy.spatial import distance


def make_spatial_weights(r):
    W = 1. / distance.squareform(distance.pdist(r))
    for i in range(W.shape[0]):
        W[i, i] = 0
    W /= np.sum(W)  # enforce unitization condition
    return W


def get_spatial_correlation(x, y, W):
    x_norm = (x - np.mean(x)) / np.sqrt(np.var(x))
    y_norm = (y - np.mean(y)) / np.sqrt(np.var(y))
    Rc = np.matmul(np.matmul(x_norm.transpose(), W), y_norm)
    return Rc


# read the cad file
os.chdir(r"/Volumes/GoogleDrive/My Drive/Documents/Research/easternStatePenitentiary/2020_1_28_files/allOfIt")
cadFile = r"2020_01_24_DRAFT_West_Wall_mkr.dxf"
doc = ezdxf.readfile(cadFile)

# record all entities in modelspace
msp = doc.modelspace()
all_objs = [e for e in msp]
layers = sorted(set([x.dxf.layer for x in all_objs]))  # set is a list of unique items

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

distMat = np.ones([len(xRand), len(layers)], dtype=np.float) * np.Inf

# broken layers: 0, 1, 2, 3, 9
# (found this by taking sum of truthMat over axis 0, all zero means no points outside that MultiPolygon)
# suspect there are open polygons being read into shapely

for i, point in tqdm(enumerate(pointList), total=len(pointList)):
    for j, layer in enumerate(layers):
        mp = layerDict[layer]
        if len(mp) == 0:
            continue
        d = mp.distance(point)
        distMat[i, j] = d

thresh = 100.
truthMat = distMat < thresh

plt.scatter(xRand, yRand, s=1, c=truthMat[:, 25], vmin=0, vmax=1)
plt.show()

# regular fa
fa = FactorAnalyzer()
numFactors = 6
df = pd.DataFrame(truthMat, columns=layers)
unnecessaryColumns = [layers[x] for x in [0, 1, 2, 3, 9]]
df.drop(unnecessaryColumns, axis=1, inplace=True)
df.dropna(inplace=True)
fa.analyze(df, numFactors, rotation=None)
L = np.array(fa.loadings)
headings = list(fa.loadings.transpose().keys())
factor_threshold = 0.25
for i, factor in enumerate(L.transpose()):
    descending = np.argsort(np.abs(factor))[::-1]
    contributions = [(np.round(factor[x], 2), headings[x]) for x in descending if np.abs(factor[x]) > factor_threshold]
    print('Factor %d:' % (i + 1), contributions)
