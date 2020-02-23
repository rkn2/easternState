#!/usr/bin/env python
# in terminal pip install ezdxf==0.6.2
import ezdxf
import os
import math
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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


def makeScree(fa):
    ev, v = fa.get_eigenvalues()
    plt.scatter(range(1, df.shape[1] + 1), np.log(ev))
    plt.plot(range(1, df.shape[1] + 1), np.log(ev))
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()


# when you return a bunch of things, should be class; for now funct
def get_data(cadFile, num_samples=1000):  # cadFile is complete path
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
    xRand = np.random.uniform(min_max[0, 0], min_max[1, 0], num_samples)  # rerun for pred
    yRand = np.random.uniform(min_max[0, 1], min_max[1, 1], num_samples)  # rerun for pred
    pointList = [Point(xRand[i], yRand[i]) for i in range(num_samples)]  # rerun for pred

    distMat = np.ones([len(xRand), len(layers)], dtype=np.float) * np.Inf

    for i, point in tqdm(enumerate(pointList), total=len(pointList)):
        for j, layer in enumerate(layers):
            mp = layerDict[layer]
            if len(mp) == 0:
                continue
            d = mp.distance(point)
            distMat[i, j] = d
    return distMat, xRand, yRand, layers


# TODO instead of distMat, return df in dict or something, each column goes in dict entry

def specific_combine(distMat, layers, newLayers):
    for layer in layers:
        for new in newLayers:
            if layer not in newLayers and bool([ele for ele in newLayers if (ele in layer)]) == False:
                newLayers.append(layer)

    newLayers.append('E-METL-T1')
    newLayers.append('E-METL-T4')

    nDistMat = np.ones([distMat.shape[0], len(newLayers)]) * np.inf
    for i, entry in enumerate(newLayers):
        these_layers = []
        for j, layer in enumerate(layers):
            if entry in layer:
                if 'E-METL' in layer:
                    if layer[-1] == '2' or '4':
                        these_layers.append(j)
                else:
                    these_layers.append(j)
        if len(these_layers) > 0:
            nDistMat[:, i] = np.min(distMat[:, these_layers], axis=1)
        # print('the layers are: ' + str(these_layers))

    # truthMat = np.exp(-nDistMat ** 2 / 8e4)

    return nDistMat, newLayers


def main():
    # read the cad file
    cadPath = r"/Volumes/GoogleDrive/My Drive/Documents/Research/easternStatePenitentiary/2020_1_28_files/allOfIt/"
    #specWall = r"2020_01_24_DRAFT_West_Wall_mkr.dxf"
    specWall = r"2020-01-24 - DRAFT North Wall.dxf"
    cadFile = cadPath + specWall
    distMat, xRand, yRand, layers = get_data(cadFile, num_samples=10000)

    # to get index of certain layer
    # indices = [i for i, s in enumerate(layers) if 'E-METL-T1' in s]

    # specific combine
    newLayers = ['W-MRTR-BCKP', 'W-MRTR-FNSH', 'C-REPR', 'E-METL', 'E-VEGT', 'W-STON-BULG', 'W-STON-RESET']
    nDistMat, newLayers = specific_combine(distMat, layers, newLayers)

    # to get index of certain layer
    # indices = [i for i, s in enumerate(layers) if 'E-METL-T1' in s]
    y = nDistMat[:, 4] == 0  # makes it binary
    metlD = nDistMat[:, [3, 21, 22]]  # rerun for pred
    # X = np.hstack([xRand.reshape(-1, 1), yRand.reshape(-1, 1), metlD]) #rerun for pred
    X = np.hstack([yRand.reshape(-1, 1), metlD])  # rerun for pred
    rf = RandomForestClassifier(n_estimators=500, oob_score=True)
    rf = rf.fit(X, y)
    rfO = rf.oob_score_  # if bad (< 0.7), rf cant handle this prediction
    Z = rf.predict_proba(X)  # rerun for pred
    fig = plt.figure()
    plt.scatter(xRand, yRand, s=1, c=Z[:, 1], vmin=0, vmax=1)
    plt.title('Predicted dif wall')
    plt.show()
    # rf.feature_importances_

    truthMat = nDistMat == 0
    fig = plt.figure()
    plt.scatter(xRand, yRand, s=1, c=truthMat[:, 4], vmin=0, vmax=1)
    plt.title('Truth wall')
    plt.show()


if __name__ == '__main__':
    main()

# # regular fa
# fa = FactorAnalyzer()
# numFactors = 4
# # df = pd.DataFrame(truthMat, columns=layers)
# # unnecessaryColumns = [layers[x] for x in [0, 1, 2, 3, 9]]
# df = pd.DataFrame(truthMat, columns=newLayers)
# unnecessaryColumns = [x for i, x in enumerate(newLayers) if np.max(nDistMat[:, i]) == np.Inf]
# df.drop(unnecessaryColumns, axis=1, inplace=True)
# df.dropna(inplace=True)
# fa.analyze(df, numFactors, rotation=None)
# L = np.array(fa.loadings)
# headings = list(fa.loadings.transpose().keys())
# factor_threshold = 0.4
# factors = []
# for i, factor in enumerate(L.transpose()):
#     descending = np.argsort(np.abs(factor))[::-1]
#     contributions = [(np.round(factor[x], 2), headings[x]) for x in descending if np.abs(factor[x]) > factor_threshold]
#     factors.append(contributions)
#     print('Factor %d:' % (i + 1), contributions)
#
# factor_list = []
# weight_list = []
# for i, factor in enumerate(factors):
#     factor_list.append([])
#     weight_list.append([])
#     for j, condition in enumerate(factor):
#         factor_list[i].append(condition[1])
#         weight_list[i].append(condition[0])
#
# for m, entry in enumerate(factor_list):
#     n = int(math.ceil(len(entry) / 3))
#     fig = plt.figure()
#
#     for k, val in enumerate(entry):
#         index = [i for i, s in enumerate(newLayers) if val in s]
#         ax = fig.add_subplot(n, 3, k + 1)
#         ax.scatter(xRand, yRand, s=1, c=truthMat[:, index[0]], vmin=0, vmax=1)
#         title_string = val + ' ' + str(weight_list[m][k])
#         ax.set_title(title_string)
#         # Hide x labels and tick labels for top plots and y ticks for right plots.
#         ax.label_outer()
