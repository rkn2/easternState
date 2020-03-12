#!/usr/bin/env python
# in terminal pip install ezdxf==0.6.2
import ezdxf
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point
from shapely.geometry import Polygon, MultiPolygon
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from mpl_toolkits.mplot3d import axes3d, Axes3D
# from factor_analyzer import FactorAnalyzer
import seaborn as sns
from sklearn.base import clone
from scipy.spatial import distance
import collections


def make_spatial_weights(r):
    W = 1. / distance.squareform(distance.pdist(r))
    for i in range(W.shape[0]):
        W[i, i] = 0
    W /= np.sum(W)  # enforce unitization condition
    return W


def get_spatial_correlation(x, y, W):
    x_norm = (x - np.mean(x)) / np.sqrt(np.var(x))
    y_norm = (y - np.mean(y)) / np.sqrt(np.var(y))
    rc = np.matmul(np.matmul(x_norm.transpose(), W), y_norm)
    return rc


def points_from_mp(mp, ax=None):
    pts = np.zeros([0, 3])
    for p in mp:
        r = np.array([x for x in p.exterior.coords])
        if ax is not None:
            ax.plot(*r.transpose(), marker='.')
        pts = np.vstack([pts, r])
    return pts


def get_layer_points(ch, work_area, points):
    points = points.tolist()
    if len(points) > 2:
        poly = Polygon(points)
        if not poly.exterior.is_valid:
            # return None
            if poly.convex_hull.exterior.is_valid:
                poly = poly.convex_hull
            else:
                return None
    else:
        return None
    points_3d = ch.embed(points, allow_inexact=False)
    if points_3d is None:
        return None
    poly = Polygon(points_3d)
    # if poly.area > work_area or not poly.exterior.is_valid:
    #     return None
    return poly


# def makeScree(fa):
#     ev, v = fa.get_eigenvalues()
#     plt.scatter(range(1, df.shape[1] + 1), np.log(ev))
#     plt.plot(range(1, df.shape[1] + 1), np.log(ev))
#     plt.title('Scree Plot')
#     plt.xlabel('Factors')
#     plt.ylabel('Eigenvalue')
#     plt.grid()
#     plt.show()

class coordHandler(object):
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
        if len(points) > 3:
            cent = Polygon(points).centroid
        else:
            cent = Point(np.mean(points, axis=0))
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


def points_in_circle_np(radius, x0=0, y0=0, n=20):
    theta = np.linspace(0, 2 * np.pi, n)
    points = np.vstack([np.cos(theta) * radius + x0, np.sin(theta) * radius + y0]).T
    # plt.scatter(points.T)
    return points


def get_index(name, layers):
    i = 0
    index = 0
    while i < len(layers):
        lyr = layers[i]
        if lyr == name:
            index = i
        i += 1
    return index


def show_layer_points(poly_list):
    for k in poly_list.keys():
        for polygon in poly_list[k]:
            x, y = polygon.exterior.xy
            plt.plot(x, y)


# when you return a bunch of things, should be class; for now funct
def get_data(cad_files, num_samples=1000):  # cadFile is complete path
    dist_df = None
    for file in cad_files:
        doc = ezdxf.readfile(file)
        # record all entities in model space
        msp = doc.modelspace()
        all_objs = [e for e in msp]
        layers = sorted(set([x.dxf.layer for x in all_objs]))

        bb = [x for x in all_objs if 'boundBox' in x.dxf.layer]  # This still covers new layers
        bb_pts = [np.array([x[:2] for x in y.get_points()]) for y in bb]
        bb_all = np.vstack(bb_pts)
        min_max = np.vstack([np.min(bb_all, axis=0), np.max(bb_all, axis=0)])
        work_area = np.prod(np.diff(min_max, axis=0))
        ch = coordHandler(bb_pts)
        layer_dict = {}

        # get wall wall_directions
        this_facing = None
        wall_directions = {'North': 0, 'East': 1, 'South': 2, 'West': 3}
        for k, v in wall_directions.items():
            if k in file:
                wall_position = wall_directions[k] * np.ones(num_samples)
                point_facing = wall_directions[k] * np.ones(num_samples)
        if wall_position is None:
            raise ValueError(
                'Cannot determine wall_directions from file name! (Looked for %s)' % wall_directions.keys())

        split_circ_poly = ['W-CRCK']
        accepted_lines = ['W-CRCK']
        remove_layers = []  # 'boundBox'
        open_poly_layers = collections.defaultdict(int)
        none_layer = collections.defaultdict(int)
        # todo: w-mrtr-fnsh-t2 see trello board (inverse of others)
        # 17 is biogrowth on north
        # get all the polylines in each layer
        for lyr in layers:
            layer_objs = [x for x in all_objs if x.dxf.layer == lyr]
            # 0 is lwpoly and 40 is circle
            poly_list = {'CIRC': [], 'POLY': []}
            for idx, obj in enumerate(layer_objs):
                if obj.dxftype() == 'CIRCLE':
                    center = obj.dxf.center
                    radius = obj.dxf.radius
                    points = points_in_circle_np(radius, center[0], center[1])
                elif 'get_points' in dir(obj):
                    points = np.array([y for y in obj.get_points()])[:, :2]
                else:
                    # print('Problem reading points from %d in layer %s' % (idx, lyr))
                    continue
                if obj.dxftype() == 'CIRCLE':
                    poly = get_layer_points(ch, work_area, points)
                    if poly is None:
                        print('Problem creating circle from %d in layer %s' % (idx, lyr))
                        continue
                    poly_list['CIRC'].append(poly)
                    # has one v for k now
                else:
                    poly = get_layer_points(ch, work_area, points)
                    if poly:  # and poly.is_valid and poly.exterior.is_closed:
                        poly_list['POLY'].append(poly)
                        continue
                    #
                    has_exterior = 'exterior' in dir(obj)
                    if not has_exterior or (has_exterior and not obj.exterior.is_closed):
                        ext_points = np.vstack([points - np.array([1e-2, 0]), points[::-1] + np.array([1e-2, 0])])
                        poly = get_layer_points(ch, work_area, ext_points)
                        if poly:  # and poly.is_valid and poly.exterior.is_closed:
                            poly_list['POLY'].append(poly)
                            continue
                    else:
                        print('dont know what to do with %d in layer %s' % (idx, lyr))
                    if poly is None:
                        none_layer[lyr] += 1
                        if none_layer[lyr] > 1 and 'ANNO' not in lyr:
                            print('got anomalous None poly from %s - %d' % (lyr, idx))

            if lyr in split_circ_poly:
                for k, v in poly_list.items():
                    layer_dict['%s-%s' % (lyr, k)] = v
            else:
                layer_dict[lyr] = poly_list['CIRC'] + poly_list['POLY']

        # update the layers list to reflect split layers
        for lyr in layers:
            if lyr in split_circ_poly:
                layers.remove(lyr)
                for k in poly_list.keys():
                    layers.append('%s-%s' % (lyr, k))

        _ = [layer_dict.pop(rm_file) for rm_file in remove_layers if rm_file in layer_dict]
        _ = [layers.remove(rm_file) for rm_file in remove_layers if rm_file in layers]

        # pick random points and convert them to 3d
        point_list = []
        while len(point_list) < num_samples:
            obj = np.random.uniform(min_max[0, 0], min_max[1, 0])
            y = np.random.uniform(min_max[0, 1], min_max[1, 1])
            r = ch.embed([[obj, y]], allow_inexact=False)
            if r is None:
                continue
            else:
                point_list.append(r[0])
        point_list = np.array(point_list)

        dist_mat = np.ones([num_samples, len(layer_dict.keys())], dtype=np.float) * np.Inf

        z_thresh = ch.bounds[2, 1] - ch.bounds[2, 0]
        for i, r in tqdm(enumerate(point_list), total=len(point_list)):
            for j, layer in enumerate(layer_dict.keys()):
                mp = layer_dict[layer]
                # mp_pts = points_from_mp(mp, ax=ax)
                mp_pts = points_from_mp(mp, ax=None)
                z_levels = np.unique(mp_pts[:, 2])
                if len(mp) == 0:
                    continue
                r_dists = []
                for p in mp:
                    rd = p.distance(Point(r))
                    if rd < z_thresh and (r[2] > z_levels.max() or r[2] < z_levels.min()):
                        rd = np.min(np.abs(z_levels - r[2]))
                    r_dists.append(rd)
                    if rd == 0:
                        break
                dist_mat[i, j] = np.min(r_dists)

        # fig = plt.figure(figsize=(10, 6))
        # ax = [None]
        # ax[0] = fig.add_subplot(111, projection='3d')
        # ax[0].scatter(point_list.T[0], point_list.T[1], point_list.T[2], s=20, c=point_facing.flatten(), vmin=-1, vmax=3)

        # todo: fix inside vs outside
        inside_idx = np.argwhere(point_list[:, 2] == ch.bounds[2, 0])
        point_facing[inside_idx] = (wall_position[inside_idx] + 2) % 4
        outside_idx = np.argwhere(point_list[:, 2] == ch.bounds[2, 1])
        point_facing[outside_idx] = wall_position[outside_idx]
        coping_idx = np.argwhere((point_list[:, 2] > ch.bounds[2, 0]) * (point_list[:, 2] < ch.bounds[2, 1]))
        point_facing[coping_idx] = -1

        dist_dict = {layer: dist_mat[:, j] for j, layer in enumerate(layers)}
        dist_df_single = pd.DataFrame.from_dict(dist_dict)
        dist_df_single.insert(0, 'wall_position', wall_position)
        dist_df_single.insert(0, 'point_facing', point_facing)
        dist_df_single.insert(0, 'z', point_list[:, 2])
        dist_df_single.insert(0, 'y', point_list[:, 1])
        dist_df_single.insert(0, 'x', point_list[:, 0])

        if dist_df is None:
            dist_df = dist_df_single
        else:
            # dist_df = pd.merge(left=dist_df, right=dist_df_single, how='outer')
            dist_df = pd.merge(left=dist_df, right=dist_df_single, how='inner')

        # remove NaNs from final data frame
        dist_df[np.isnan(dist_df)] = np.Inf

        # visualize results
        # fig, ax = plt.subplots(1, 1, fig_size=(10, 4))
        # ax.scatter(xRand, yRand, c=dist_df['E-VEGT-BIOGRW'])
        # ax.set_xlim(min_max[:, 0])
        # ax.set_ylim(min_max[:, 1])
        # ax.set_aspect('equal')

        return dist_df


def specific_combine(dist_df, new_layers):
    layers = dist_df.columns
    for layer in layers:
        # for new in newLayers:
        if layer not in new_layers and bool([ele for ele in new_layers if (ele in layer)]) == False:
            new_layers.append(layer)

    new_layers.append('E-METL-T1')
    new_layers.append('E-METL-T4')

    num_points = None
    for layer in layers:
        if num_points is None:
            num_points = len(dist_df[layer])
        else:
            if len(dist_df[layer]) != num_points:
                raise ValueError('All layers must have same number of points!')

    n_dist_mat = np.ones([num_points, len(new_layers)]) * np.inf
    for i, entry in enumerate(new_layers):
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
            n_dist_mat[:, i] = np.min(these_dists, axis=1)
        # print('the layers are: ' + str(these_layers))

    # truthMat = np.exp(-n_dist_mat ** 2 / 8e4)

    return n_dist_mat, new_layers


def cross_corr(dist_df, num_vars=5):
    new_dist_df = dist_df[['E-VEGT-GROWIES', 'E-VEGT-BIOGRW', 'y', 'C-CRCK', 'W-SURF-STAIN-T1']]
    new_dist_df.corr()
    colormap = plt.cm.RdBu
    plt.figure(figsize=(15, 10))
    # plt.title(u'6 hours', y=1.05, size=16)

    mask = np.zeros_like(new_dist_df.corr())
    mask[np.triu_indices_from(mask)] = True

    svm = sns.heatmap(new_dist_df.corr(), mask=mask, linewidths=0.1, vmax=1.0,
                      square=True, cmap=colormap, linecolor='white', annot=True)


def drop_col_feat_imp(model, X_train, y_train, certain_layer, new_layers, random_state=42):
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    # list for storing feature importances
    importances = []

    n_dist_mat_df = pd.DataFrame(data=X_train, columns=new_layers + ['random'])
    X_train_df = pd.DataFrame(data=X_train, columns=new_layers + ['random'])

    # file1 = open(r"test.txt", "w+")
    # text = []
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train_df.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train_df.drop(col, axis=1), y_train)
        drop_col_score = model_clone.score(X_train_df.drop(col, axis=1), y_train)
        importances.append(benchmark_score - drop_col_score)
        # text.append(str(col) + ' : ' + str(benchmark_score - drop_col_score) + '\n')
    importances_df = pd.DataFrame(importances, X_train_df.columns)
    ax = importances_df.plot.barh(rot=0, figsize=(12, 10))
    fig = ax.get_figure()
    plt.show(block=False)
    rand_mag = np.abs(importances_df.values[-1])
    ax.plot(rand_mag * np.ones(2), ax.get_ylim(), 'k--')
    ax.plot(-rand_mag * np.ones(2), ax.get_ylim(), 'k--')
    plt.close(fig)
    file_name = str(certain_layer) + '.pdf'
    ax.figure.savefig(file_name)

    # file1.write(text)
    # file1.close()
    return importances_df


def pred_locations(x_rand, y_rand, z_rand, rf, X, y, certain_layer):
    # Z = rf.predict(X)
    Z = rf.predict_proba(X)[:, 1]  # rerun for pred (since just x, it doesnt use y (veggie))
    fig = plt.figure(figsize=(10, 6))
    ax = [None, None]
    ax[0] = fig.add_subplot(211, projection='3d')
    ax[0].scatter(x_rand, y_rand, z_rand, s=20, c=Z, vmin=0, vmax=1)
    ax[0].set_title('Predicted wall for ' + certain_layer)
    # compare to ground truth
    ax[1] = fig.add_subplot(212, projection='3d')
    ax[1].scatter(x_rand, y_rand, z_rand, s=20, c=y, vmin=0)  # , vmax=1)
    ax[1].set_title('Ground truth for ' + certain_layer)
    file_name = str(certain_layer) + '_PredVsGround.pdf'
    fig.savefig(file_name)


def main(thresh=10):
    # read the cad file
    cad_path = r"/Volumes/GoogleDrive/My Drive/Documents/Research/easternStatePenitentiary/2020_2_7_files/CAD_Progress/"
    walls = [r'2020 03 11 - et - DRAFT West Wall_BN.dxf', r'2020 03 11 - et - DRAFT North Wall_BN.dxf',
             r'2020 03 11 - et - DRAFT South Wall_BN.dxf', r'2020 03 11 - et - DRAFT East Wall_BN.dxf']
    cad_files = [cad_path + walls[itup[0]] for itup in enumerate(walls)]
    dist_df = get_data(cad_files, num_samples=1000)
    layers = dist_df.columns

    for certainLayer in tqdm(layers, total=len(layers)):
        # to get index of certain layer
        prohibited_layers = ['0', certainLayer, 'W-STON-DELM', 'W-STON-STRAT', 'E-METL-T3', 'W-STON-RESET-T4']
        layer_order = sorted(dist_df.keys())
        n_dist_mat = np.array([dist_df[k].values for k in layer_order if k not in prohibited_layers])
        # new_layers = layers
        new_layers = [k for k in layer_order if k not in prohibited_layers]

        rand_column = np.random.randint(2, size=n_dist_mat.shape[1])
        n_dist_mat = np.vstack([n_dist_mat, rand_column])

        X = n_dist_mat.transpose()
        y = dist_df[certainLayer].values <= thresh

        # # transform to (0, 1)
        # n_dist_mat = np.exp(-n_dist_mat/1e3)

        x_rand = np.array(dist_df['x'])
        y_rand = np.array(dist_df['y'])
        z_rand = np.array(dist_df['z'])
        X[np.isinf(X)] = 1e12

        # set aside some data for testing the model later
        test_fraction = 0.25
        test_number = int(test_fraction * X.shape[0])
        test_idx = np.random.choice(X.shape[0], test_number, replace=False)
        test_mat = np.zeros(X.shape[0], dtype=np.bool)
        test_mat[test_idx] = True
        train_mat = ~test_mat
        print('working on %s' % certainLayer)
        if np.sum(y[train_mat]) == 0 or np.sum(~y[train_mat]) == 0:
            print('skipping %s' % certainLayer)
            continue

        # train the random forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                    oob_score=True)  # n_estimators=500, WANT LOWER NUMBER --> MORE GENERIC
        rf = rf.fit(X[train_mat], y[train_mat])  # train only on some part of the data

        # eval model on test data
        # print('OOB score %f' % rf.oob_score_)  # if bad (< 0.7), rf cant handle this prediction
        # print('score on test data %f' % rf.score(X[test_mat], y[test_mat]))  # get the score on unseen data
        # print('layer contribution to RF in descending order:')
        # for x in np.argsort(rf.feature_importances_)[::-1]:
        #     # print('%3d : %10f : %s' % (idx[x], rf.feature_importances_[x], new_layers[idx[x]]))
        #     print('%3d : %10f : %s' % (x, rf.feature_importances_[x], new_layers[x]))

        # figure out important features
        drop_col_feat_imp(rf, X, y, certainLayer, new_layers)

        # make predictions and plot them
        pred_locations(x_rand, y_rand, z_rand, rf, X, y, certainLayer)


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
