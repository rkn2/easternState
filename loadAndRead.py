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
from mpl_toolkits.mplot3d import Axes3D
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


def points_from_mp(mp, ax=None):
    pts = np.zeros([0, 3])
    for p in mp:
        r = np.array([x for x in p.exterior.coords])
        if ax is not None:
            ax.plot(*r.transpose(), marker='.')
        pts = np.vstack([pts, r])
    return pts

# def makeScree(fa):
#     ev, v = fa.get_eigenvalues()
#     plt.scatter(range(1, df.shape[1] + 1), np.log(ev))
#     plt.plot(range(1, df.shape[1] + 1), np.log(ev))
#     plt.title('Scree Plot')
#     plt.xlabel('Factors')
#     plt.ylabel('Eigenvalue')
#     plt.grid()
#     plt.show()

class coordinate_handler(object):
    def __init__(self, bb_pts):
        # get vertical sequence of wall segments
        vert_order = np.argsort([np.min(x[:, 1]) for x in bb_pts])
        # bottom-most is correct orientation, middle is coping, top-most is up-down mirrored
        self.segments = [Polygon(bb_pts[v]) for v in vert_order]
        self.all_segments = MultiPolygon(self.segments)
        # establish ranges for the transform
        self.y_range = [np.min(bb_pts[vert_order[0]][:, 1]),
                        np.max(bb_pts[vert_order[0]][:, 1])]
        self.z_range = [np.min(bb_pts[vert_order[1]][:, 1]),
                        np.max(bb_pts[vert_order[1]][:, 1])]
        # set up the transform
        self.top_box = [np.min(bb_pts[vert_order[2]][:, 1]),
                        np.max(bb_pts[vert_order[2]][:, 1])]
        # record the 3d bounding box
        self.bounds = np.array([[np.min(np.vstack(bb_pts)[:, 0]), np.max(np.vstack(bb_pts)[:, 0])],  # x range
                                self.y_range, self.z_range])
    def xform_front(self, p):
        return [p[0], p[1], self.z_range[0]]
    def xform_back(self, p):
        return [p[0], self.top_box[1] - p[1] + self.y_range[0], self.z_range[1]]
    def xform_top(self, p):
        return [p[0], self.y_range[1], p[1]]
    def embed(self, points, allow_inexact=False):
        # points = [p for p in raw_points if self.all_segments.contains(Point(p))]
        # if len(points) < 3:
        #     return None
        cent = np.mean(points, axis=0)
        segment_distance = [x.distance(Point(cent)) for x in self.segments]
        best_segment = np.argmin(segment_distance)
        if not allow_inexact and segment_distance[best_segment] > 0:
            return None
        if best_segment == 2:
            return [self.xform_back(p) for p in points]
        elif best_segment == 1:
            return [self.xform_top(p) for p in points]
        elif best_segment == 0:
            return [self.xform_front(p) for p in points]


# when you return a bunch of things, should be class; for now funct
def get_data(cadFiles, num_samples=1000):  # cadFile is complete path
    dist_df = None
    for file in cadFiles:
        doc = ezdxf.readfile(file)
        # record all entities in modelspace
        msp = doc.modelspace()
        all_objs = [e for e in msp]
        layers = sorted(set([x.dxf.layer for x in all_objs]))  # set is a list of unique items

        bb = [x for x in all_objs if 'boundBox' in x.dxf.layer]
        bb_pts = [np.array([x[:2] for x in y.get_points()]) for y in bb]
        bottom_left = [np.min(x, axis=0) for x in bb_pts]

        bb_all = np.vstack(bb_pts)
        min_max = np.vstack([np.min(bb_all, axis=0), np.max(bb_all, axis=0)])
        work_area = np.prod(np.diff(min_max, axis=0))

        # min_max = np.vstack([np.min(front, axis=0), np.max(front, axis=0)])

        # min_max = np.array([np.Inf * np.ones(2), -np.Inf * np.ones(2)])
        # for x in all_objs:
        #     try:
        #         p = np.array([y for y in x.get_points()])[:, :2]
        #         min_max[0] = np.min([min_max[0], np.min(p, axis=0)], axis=0)
        #         min_max[1] = np.max([min_max[1], np.max(p, axis=0)], axis=0)
        #     except AttributeError:
        #         pass

        # get wall facing
        this_facing = None
        facing = {'North': 0, 'East': 1, 'South': 2, 'West': 3}
        opposing = {'North': 2, 'East': 3, 'South': 0, 'West': 1}
        for k, v in facing.items():
            if k in file:
                wall_facing = facing[k] * np.ones(num_samples)
                wall_opposite = opposing[k]
                point_facing = -np.ones(num_samples)
        if wall_facing is None:
            raise ValueError('Cannot determine facing from file name! (Looked for %s)' % facing.keys())

        # for visualizing the 3d wall
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        ch = coordinate_handler(bb_pts)

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
                    if not poly.is_valid:
                        continue
                else:
                    continue
                points_3d = ch.embed(points, allow_inexact=False)
                if points_3d is None:
                    continue
                poly = Polygon(points_3d)
                # ax.plot(*np.array(points_3d).transpose())
                if poly.area < work_area:
                    polyList.append(poly)
                else:
                    raise ValueError('A polygon is larger than the work area!')
            # mp = unary_union(polyList)
            # mp = MultiPolygon(polyList)
            # if not mp.is_valid:
            #     raise RuntimeError
            # layerDict.update({lyr: MultiPolygon(polyList)})
            layerDict[lyr] = polyList

        if 'boundBox' in layerDict:
            del layerDict['boundBox']
        if 'boundBox' in layers:
            layers.remove('boundBox')

        # pick random points and convert them to 3d
        pointList = []
        while len(pointList) < num_samples:
            x = np.random.uniform(min_max[0, 0], min_max[1, 0])
            y = np.random.uniform(min_max[0, 1], min_max[1, 1])
            r = ch.embed([[x, y]], allow_inexact=False)
            if r is None:
                continue
            else:
                pointList.append(r[0])
        pointList = np.array(pointList)

        # xRand = np.random.uniform(min_max[0, 0], min_max[1, 0], num_samples)  # rerun for pred
        # yRand = np.random.uniform(min_max[0, 1], min_max[1, 1], num_samples)  # rerun for pred
        # pointList = [Point(xRand[i], yRand[i]) for i in range(num_samples)]  # rerun for pred

        distMat = np.ones([num_samples, len(layers)], dtype=np.float) * np.Inf

        for i, r in tqdm(enumerate(pointList), total=len(pointList)):
            # todo: create a point_facing that specifies the direction of the normal
            # point_facing[i] =
            for j, layer in enumerate(layers):
                mp = layerDict[layer]
                # mp_pts = points_from_mp(mp, ax=ax)
                mp_pts = points_from_mp(mp, ax=None)
                z_levels = np.unique(mp_pts[:, 2])
                if len(mp) == 0:
                    continue
                r_dists = [p.distance(Point(r)) for p in mp]
                d = np.min(r_dists)
                if d == 0 and r[2] not in z_levels:
                    d = np.min(np.abs(z_levels - r[2]))
                distMat[i, j] = d

        dist_dict = {layer: distMat[:, j] for j, layer in enumerate(layers)}
        dist_df_single = pd.DataFrame.from_dict(dist_dict)
        dist_df_single.insert(0, 'wall_facing', wall_facing)
        # dist_df_single.insert(0, 'point_facing', point_facing)
        dist_df_single.insert(0, 'z', pointList[:, 2])
        dist_df_single.insert(0, 'y', pointList[:, 1])
        dist_df_single.insert(0, 'x', pointList[:, 0])
        if dist_df is None:
            dist_df = dist_df_single
        else:
            # dist_df = pd.merge(left=dist_df, right=dist_df_single, how='outer')
            dist_df = pd.merge(left=dist_df, right=dist_df_single, how='inner')

        # remove NaNs from final dataframe
        dist_df[np.isnan(dist_df)] = np.Inf

        # visualize results
        # fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        # ax.scatter(xRand, yRand, c=dist_df['E-VEGT-BIOGRW'])
        # ax.set_xlim(min_max[:, 0])
        # ax.set_ylim(min_max[:, 1])
        # ax.set_aspect('equal')

        return dist_df


def specific_combine(dist_df, newLayers):
    layers = dist_df.columns
    for layer in layers:
        # for new in newLayers:
        if layer not in newLayers and bool([ele for ele in newLayers if (ele in layer)]) == False:
            newLayers.append(layer)

    newLayers.append('E-METL-T1')
    newLayers.append('E-METL-T4')

    num_points = None
    for layer in layers:
        if num_points is None:
            num_points = len(dist_df[layer])
        else:
            if len(dist_df[layer]) != num_points:
                raise ValueError('All layers must have same number of points!')

    nDistMat = np.ones([num_points, len(newLayers)]) * np.inf
    for i, entry in enumerate(newLayers):
        these_layers = []
        for j, layer in enumerate(layers):
            if entry in layer:
                if 'E-METL' in layer:
                    if layer[-1] == '2' or '4':
                        these_layers.append(layer)
                else:
                    these_layers.append(layer)
        if len(these_layers) > 0:
            these_dists = np.vstack([dist_df[layer] for layer in these_layers]).transpose()
            nDistMat[:, i] = np.min(these_dists, axis=1)
        # print('the layers are: ' + str(these_layers))

    # truthMat = np.exp(-nDistMat ** 2 / 8e4)

    return nDistMat, newLayers


def main():
    # read the cad file
    cadPath = r"/Volumes/GoogleDrive/My Drive/Documents/Research/easternStatePenitentiary/2020_1_28_files/allOfIt/"
    walls = [r'2020 02 06 - DRAFT West Wall mkr-et v02_BN.dxf', r'2020-01-24 - DRAFT North Wall_BN.dxf',
             r'2020-02-06 - DRAFT South Wall_BN.dxf']
    cadFiles = [cadPath + walls[itup[0]] for itup in enumerate(walls)]
    dist_df = get_data(cadFiles, num_samples=1000)

    # to get index of certain layer
    # indices = [i for i, s in enumerate(layers) if 'E-METL-T1' in s]

    # # specific combine
    # newLayers = ['W-MRTR-BCKP', 'W-MRTR-FNSH', 'C-REPR', 'E-METL', 'E-VEGT', 'W-STON-BULG', 'W-STON-RESET']
    # nDistMat, newLayers = specific_combine(dist_df, newLayers)
    #
    # to get index of certain layer
    certainLayer = 'E-VEGT-BIOGRW'
    prohibited_layers = ['0', certainLayer, 'E-VEGT-GROWIES']
    layer_order = sorted(dist_df.keys())
    nDistMat = np.array([dist_df[k].values for k in layer_order if k not in prohibited_layers])
    # layers = dist_df.columns
    # newLayers = layers
    newLayers = [k for k in layer_order if k not in prohibited_layers]

    X = nDistMat.transpose()
    y = dist_df[certainLayer].values == 0

    # # transform to (0, 1)
    # nDistMat = np.exp(-nDistMat/1e3)

    xRand = np.array(dist_df['x'])
    yRand = np.array(dist_df['y'])
    zRand = np.array(dist_df['z'])

    nDistMat[np.isinf(nDistMat)] = 1e12

    # indices = [i for i, s in enumerate(layers) if 'E-METL-T1' in s]

    # interest_idx = [i for i, layer in enumerate(newLayers) if certainLayer in layer]
    # y = nDistMat[:, interest_idx] == 0  # makes it binary
    # # metlD = nDistMat[:, [3, 21, 22]]
    # # X = np.hstack([xRand.reshape(-1, 1), yRand.reshape(-1, 1), metlD])
    # disallowed_layers = [certainLayer, 'x']
    # idx = [i for i, layer in enumerate(newLayers) if np.all([y not in layer for y in disallowed_layers])]
    # X = nDistMat[:, idx]

    # set aside some data for testing the model later
    test_fraction = 0.25
    test_number = int(test_fraction * X.shape[0])
    test_idx = np.random.choice(X.shape[0], test_number, replace=False)
    test_mat = np.zeros(X.shape[0], dtype=np.bool)
    test_mat[test_idx] = True
    train_mat = ~test_mat

    # train the random forest
    rf = RandomForestClassifier(oob_score=True) #n_estimators=500, WANT LOWER NUMBER --> MORE GENERIC
    rf = rf.fit(X[train_mat], y[train_mat])  # train only on some part of the data

    # eval model on test data
    print('OOB score %f' % rf.oob_score_)  # if bad (< 0.7), rf cant handle this prediction
    print('score on test data %f' % rf.score(X[test_mat], y[test_mat]))  # get the score on unseen data
    print('layer contribution to RF in descending order:')
    for x in np.argsort(rf.feature_importances_)[::-1]:
        # print('%3d : %10f : %s' % (idx[x], rf.feature_importances_[x], newLayers[idx[x]]))
        print('%3d : %10f : %s' % (x, rf.feature_importances_[x], newLayers[x]))

    # make predictions and plot them
    # Z = rf.predict(X)
    Z = rf.predict_proba(X)[:, 1]  # rerun for pred (since just x, it doesnt use y (veggie))
    fig = plt.figure(figsize=(10, 6))
    ax = [None, None]
    ax[0] = fig.add_subplot(211, projection='3d')
    ax[0].scatter(xRand, yRand, zRand, s=20, c=Z, vmin=0, vmax=1)
    ax[0].set_title('Predicted wall')
    # compare to ground truth
    ax[1] = fig.add_subplot(212, projection='3d')
    ax[1].scatter(xRand, yRand, zRand, s=20, c=y, vmin=0)  # , vmax=1)
    ax[1].set_title(certainLayer)
    # plt.show()
    # fig = plt.figure()
    # plt.scatter(xRand, yRand, s=1, c=nDistMat[:, 20], vmin=0)  # , vmax=1)
    # plt.title('wsurf')
    # plt.show()

# testing
# test = dist_df['W-SURF-COAT']
# test = test[:1000]
# plt.scatter(xRand, yRand, s=1, c=test, vmin=0)  # , vmax=1)

# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(dist_df['x'], dist_df['y'], dist_df['z'],
#            s=20, c=(dist_df['E-VEGT-BIOGRW'] == 0), vmin=0, vmax=1)

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
