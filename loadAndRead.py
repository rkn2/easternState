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
from factor_analyzer import FactorAnalyzer
import seaborn as sns
from sklearn.base import clone
from scipy.spatial import distance
import collections
import os
import glob
import multiprocessing as mproc


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
    points_3d = ch.embed(points, use_hull=True, allow_inexact=False)
    if points_3d is None:
        return None
    poly = Polygon(points_3d)
    # if poly.area > work_area or not poly.exterior.is_valid:
    #     return None
    return poly


def makeScree(fa, dist_df):
    ev, v = fa.get_eigenvalues()
    fig, ax = plt.subplots(1, 1)
    ax.scatter(range(1, dist_df.shape[1] + 1), np.log(ev))
    ax.plot(range(1, dist_df.shape[1] + 1), np.log(ev))
    ax.set_title('Scree Plot')
    ax.set_xlabel('Factors')
    ax.set_ylabel('Eigenvalue')
    ax.grid()

    return fig


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

    def embed(self, points, use_hull=False, allow_inexact=False):
        # points = [p for p in raw_points if self.all_segments.contains(Point(p))]
        # if len(points) < 3:
        #     return None
        if len(points) > 3:
            cent = Polygon(points).centroid
        else:
            cent = Point(np.mean(points, axis=0))
        if use_hull:
            segment_distance = [x.convex_hull.distance(Point(cent)) for x in self.segments]
        else:
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
        # aggregated_layers = ['C-CRCK', 'C-HANG', 'C-REPR', 'E-METL', 'E-VEGT', 'W-CRCK', 'W-MRTR-BCKP', 'W-MRTR-FNSH',
        #                      'W-STON-RESET', 'W-SURF-STAIN']
        remove_layers = []  # 'boundBox'
        open_poly_layers = collections.defaultdict(int)
        none_layer = collections.defaultdict(int)

        # get all the polylines in each layer
        for lyr in layers:
            layer_objs = [x for x in all_objs if x.dxf.layer == lyr]
            # 0 is lwpoly and 40 is circle
            poly_list = {'CIRC': [], 'POLY': []}
            for idx, obj in enumerate(layer_objs):
                if obj.dxftype() == 'LINE':
                    continue
                elif obj.dxftype() == 'CIRCLE':
                    center = obj.dxf.center
                    radius = obj.dxf.radius
                    points = points_in_circle_np(radius, center[0], center[1])
                elif 'get_points' in dir(obj):
                    points = np.array([y for y in obj.get_points()])[:, :2]
                    if len(points) < 3:
                        continue
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
            #
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
        point_3d_list = []
        point_2d_list = []
        while len(point_3d_list) < num_samples:
            x = np.random.uniform(min_max[0, 0], min_max[1, 0])
            y = np.random.uniform(min_max[0, 1], min_max[1, 1])
            r = ch.embed([[x, y]], use_hull=False, allow_inexact=False)
            if r is None:
                continue
            else:
                point_3d_list.append(r[0])
                point_2d_list.append([x, y])

        point_3d_list = np.array(point_3d_list)
        point_2d_list = np.array(point_2d_list)

        dist_mat = np.ones([num_samples, len(layer_dict.keys())], dtype=np.float) * np.Inf

        z_thresh = ch.bounds[2, 1] - ch.bounds[2, 0]
        for i, r in tqdm(enumerate(point_3d_list), total=len(point_3d_list)):
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

        inside_idx = np.argwhere(point_3d_list[:, 2] == ch.bounds[2, 0])
        point_facing[inside_idx] = (wall_position[inside_idx] + 2) % 4
        outside_idx = np.argwhere(point_3d_list[:, 2] == ch.bounds[2, 1])
        point_facing[outside_idx] = wall_position[outside_idx]
        coping_idx = np.argwhere((point_3d_list[:, 2] > ch.bounds[2, 0]) * (point_3d_list[:, 2] < ch.bounds[2, 1]))
        point_facing[coping_idx] = -1

        # dist_dict = {layer: dist_mat[:, j] for j, layer in enumerate(layers)}
        dist_dict = {layer: dist_mat[:, j] for j, layer in enumerate(layer_dict.keys())}
        dist_df_single = pd.DataFrame.from_dict(dist_dict)
        dist_df_single.insert(0, 'wall_position', wall_position)
        dist_df_single.insert(0, 'point_facing', point_facing)
        dist_df_single.insert(0, 'y_orig', point_2d_list[:, 1])
        dist_df_single.insert(0, 'x_orig', point_2d_list[:, 0])
        dist_df_single.insert(0, 'z', point_3d_list[:, 2])
        dist_df_single.insert(0, 'y', point_3d_list[:, 1])
        dist_df_single.insert(0, 'x', point_3d_list[:, 0])

        if dist_df is None:
            dist_df = dist_df_single
        else:
            dist_df_single.index += len(dist_df)
            dist_df = pd.merge(left=dist_df, right=dist_df_single, how='outer')  # keep all columns
            # dist_df = pd.merge(left=dist_df, right=dist_df_single, how='inner')  # keep only common columns

        # remove NaNs from final data frame
        dist_df[np.isnan(dist_df)] = np.Inf

        # discount manual w-mrtr-fnsh-t2
        mrtr = np.zeros(num_samples)
        anti_mortar_layers = ['W-MRTR-BCKP-T1', 'W-MRTR-BCKP-T2', 'W-MRTR-BCKP-T3',
                              'W-MRTR-FNSH-T1', 'W-MRTR-OPEN', 'W-STON-BULG-T2', 'W-STON-BULG-T3',
                              'W-STON-RESET-T1', 'W-STON-RESET-T2', 'W-STON-RESET-T3',
                              'W-SURF-RENDR', 'boundBox_C']
        for i in range(len(mrtr)):
            # if its not 0 in any of those layers and it is not in the coping, then it is this
            is_anti = 1
            for nlyr in anti_mortar_layers:
                if nlyr not in dist_df.columns:
                    continue
                is_anti *= dist_df[nlyr][i]
            mrtr[i] = float(is_anti > 0)
        dist_df['W-MRTR-FNSH-T2'] = mrtr

        # visualize results
        # fig, ax = plt.subplots(1, 1, fig_size=(10, 4))
        # ax.scatter(xRand, yRand, c=dist_df['E-VEGT-BIOGRW'])
        # ax.set_xlim(min_max[:, 0])
        # ax.set_ylim(min_max[:, 1])
        # ax.set_aspect('equal')

        return layer_dict, dist_df


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


def importance_corr(dist_df, certainLayer, importances_df, max_features=None):
    #
    valid = importances_df > np.abs(importances_df.T['random'][0])
    valid_cols = [c for c in valid.T.columns if valid.T[c][0]]
    valid_imp = [importances_df[0][c] for c in valid_cols]
    sorted_cols = [valid_cols[x] for x in np.argsort(valid_imp)[::-1]]
    if max_features is not None:
        sorted_cols = sorted_cols[0:min(len(sorted_cols), max_features)]
    columns = [certainLayer] + sorted_cols
    loc_dist_df = dist_df[columns]
    #
    xcorr = loc_dist_df.corr()
    colormap = plt.cm.RdBu_r
    mask = np.zeros_like(xcorr)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(1, 1, figsize=(4, max(4, len(loc_dist_df.columns) / 5)))
    # annot = importances_df.T[columns[1:]].T
    subax = sns.heatmap(xcorr[[certainLayer]][1:], ax=ax,
                        linewidths=0.1, vmin=-1.0, vmax=1.0,
                        cmap=colormap, linecolor='white',
                        annot=True, yticklabels=True)
    fig.tight_layout()

    return fig


def cross_corr(dist_df, prohibited_layers=[]):
    h_layers = []
    for lyr in dist_df.columns:
        these = []
        segments = lyr.split('-')
        for i, s in enumerate(segments):
            these.append('-'.join(segments[:i] + [s]) + '-')
        h_layers += these[:-1]
    h_layers = sorted(set(h_layers))
    valid_cols = [l for l in dist_df.columns if l not in prohibited_layers and l not in h_layers]
    loc_dist_df = dist_df[sorted(valid_cols)]

    whole_corr = loc_dist_df.corr()
    colormap = plt.cm.RdBu_r
    # mask = np.zeros_like(loc_dist_df.corr())
    # mask[np.triu_indices_from(mask)] = True
    wh = len(valid_cols) / 5
    fig, ax = plt.subplots(1, 1, figsize=(wh, wh))
    svm_plot = sns.heatmap(whole_corr, vmin=-1.0, vmax=1.0, ax=ax,
                           square=True, cmap=colormap, linecolor='k', linewidths=0.3,
                           annot=False, yticklabels=True, xticklabels=True)
    fig.tight_layout()

    return fig


def drop_one_col(work_data):
    model, col, X_train, y_train = work_data
    model_clone = clone(model)
    col_data = X_train.drop(col, axis=1)
    model_clone.fit(col_data, y_train)
    drop_col_score = model_clone.score(col_data, y_train)
    return (col, drop_col_score)


def drop_col_feat_imp(model, X_train, y_train, certain_layer, cad_path, new_layers, random_state=42):
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)

    # n_dist_mat_df = pd.DataFrame(data=X_train, columns=new_layers + ['random'])
    X_train_df = pd.DataFrame(data=X_train, columns=new_layers + ['random'])

    # parallel implementation
    p = mproc.Pool(4)
    work_data = [(model, col, X_train_df, y_train) for col in X_train_df.columns]
    results = p.map(drop_one_col, work_data)
    p.close()
    p.join()

    results_dict = {k: v for k, v in results}
    importances = [benchmark_score - results_dict[col] for col in X_train_df.columns]

    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    # importances = []  # list for storing feature importance
    # for col in tqdm(X_train_df.columns, total=len(X_train_df.columns)):
    #     model_clone = clone(model)
    #     model_clone.random_state = random_state
    #     col_data = X_train_df.drop(col, axis=1)
    #     model_clone.fit(col_data, y_train)
    #     drop_col_score = model_clone.score(col_data, y_train)
    #     importances.append(benchmark_score - drop_col_score)
    # text.append(str(col) + ' : ' + str(benchmark_score - drop_col_score) + '\n')
    importances_df = pd.DataFrame(importances, X_train_df.columns)

    return importances_df


def pred_locations_3d(x_rand, y_rand, z_rand, rf, X, y, certain_layer):
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


def pred_locations_2d(x_orig, y_orig, y, Z,
                      certain_layer, wall_name, bboxes,
                      cmap='RdBu_r', ms=1, lw=0.5):
    fig, ax = plt.subplots(3, 1, figsize=(10, 6))
    a = ax[0]
    a.scatter(x_orig, y_orig, s=ms, c=Z, vmin=0, vmax=1, cmap=cmap)
    for bb in bboxes:
        a.plot(bb[:, 0], bb[:, 1], 'k-', lw=lw)
    a.set_title('Predicted %s wall for %s' % (wall_name, certain_layer))
    a.set_aspect('equal')
    #
    a = ax[1]
    a.scatter(x_orig, y_orig, s=ms, c=y - Z, vmin=-1, vmax=1, cmap=cmap)
    for bb in bboxes:
        a.plot(bb[:, 0], bb[:, 1], 'k-', lw=lw)
    a.set_title('Difference in %s wall for %s' % (wall_name, certain_layer))
    a.set_aspect('equal')
    # compare to ground truth
    a = ax[2]
    a.scatter(x_orig, y_orig, s=ms, c=y, vmin=0, vmax=1, cmap=cmap)
    for bb in bboxes:
        a.plot(bb[:, 0], bb[:, 1], 'k-', lw=lw)
    a.set_title('Ground truth %s wall for %s' % (wall_name, certain_layer))
    a.set_aspect('equal')

    return fig


def plot_gt_2d(x_orig, y_orig, d,
               certain_layer, wall_name, bboxes,
               cmap='RdBu_r', ms=1, lw=0.5,
               ax=None, vmin=None, vmax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2.5))

    ax.scatter(x_orig, y_orig, s=ms, c=d, cmap=cmap, vmin=vmin, vmax=vmax)
    for bb in bboxes:
        ax.plot(bb[:, 0], bb[:, 1], 'k-', lw=lw)
    ax.set_title('Distance to %s for %s wall' % (certain_layer, wall_name))
    ax.set_aspect('equal')

    if ax is None:
        return fig


def compute_distances(cad_files, cad_path, num_samples=100, num_batches=1):
    # read the cad file
    # cad_path = r"/Volumes/GoogleDrive/My Drive/Documents/Research/easternStatePenitentiary/2020_3_11/"
    # walls = [r'2020-03-11 - et - DRAFT North Wall_BN.dxf',
    #          r'2020-03-11 - et - DRAFT East Wall_BN.dxf',
    #          r'2020-03-11 - et - DRAFT South Wall_BN.dxf',
    #          r'2020-03-11 - et - DRAFT West Wall_BN.dxf']
    # cad_files = [cad_path + walls[itup[0]] for itup in enumerate(walls)]

    # get the pkl path
    pkl_files = glob.glob(os.path.join(cad_path, 'dists_*.pkl'))
    pkl_idx = sorted([int(x.split('/')[-1].split('_')[1].replace('.pkl', '')) for x in pkl_files])
    if len(pkl_idx) == 0:
        idx = 0
    else:
        idx = pkl_idx[-1] + 1
    #
    k = -1
    for batch in range(num_batches):
        for cf in cad_files:
            k +=1

            if 'South' not in cf:
                continue

            # compute the distances
            layer_dict, dist_df = get_data([cf], num_samples)

            # save the distances
            dist_df.to_pickle(os.path.join(cad_path, 'dists_%05d.pkl' % (idx + k)))



def worker_task(work_data):
    cf, pkl_path, num_samples = work_data
    # compute the distances
    layer_dict, dist_df = get_data([cf], num_samples)
    # save the distances
    dist_df.to_pickle(pkl_path)
    return True


def parallel_compute_distances(cad_files, cad_path, num_samples=100, num_batches=1):
    # get the pkl path
    pkl_files = glob.glob(os.path.join(cad_path, 'dists_*.pkl'))
    pkl_idx = sorted([int(x.split('/')[-1].split('_')[1].replace('.pkl', '')) for x in pkl_files])
    if len(pkl_idx) == 0:
        idx = 0
    else:
        idx = pkl_idx[-1] + 1
    # define the worker data
    work_data = []
    k = 0
    for batch in range(num_batches):
        for cf in cad_files:
            # compute the distances
            pkl_path = os.path.join(cad_path, 'dists_%05d.pkl' % (idx+k))
            work_data.append((cf, pkl_path, num_samples))
            k += 1
    # run the Pool
    p = mproc.Pool(4)
    results = p.map(worker_task, work_data)
    p.close()
    p.join()


def run_model(thresh=10,
              plot_correlation=False,
              plot_importance=False,
              plot_wall=False,
              calculate_importance=False,
              importance_type='model'):
    if importance_type not in ['model', 'drop_col']:
        raise ValueError('importance_type must be one of "model" or "drop_col"')

    independent_layers = ['X', 'Y', 'Z', 'POINT_FACING', 'WALL_POSITION', 'E-METL-T2',
                          'E-METL-T4', 'W-STON-HOLE', 'BOUNDBOX_C', 'BOUNDBOX_I', 'BOUNDBOX_O',
                          'IMPERVIOUS', 'PERVIOUS', 'TREE', 'C-HANG', 'C-HANG-1', 'C-HANG-2',
                          'C-HANG-3', 'C-HANG-4', 'C-HANG-5', 'C-HANG-6', 'C-HANG-7',
                          'ASHLAR', 'BEHIND_PILLAR', 'WINDOW', 'U-WL', 'U-UKN', 'CUTSTONE',
                          'U-WL-DRAIN', 'U-WL-RFDRAIN', 'U-WL-RFDRAIN-OLD', 'U-SD']

    prohibited_layers = ['W-STON-DELAM', 'W-STON-STRAT-T1', 'E-METL-T3', 'W-STON-RESET-T4',
                         '0', '0-TIFF', 'A-ANNO-COLCTR', 'A-ANNO-COLNO', 'A-ANNO-CUTLINE', 'A-',
                         'A-ANNO-', 'DEFPOINTS', 'X_ORIG', 'Y_ORIG', 'W-STON-STRAT-T2',
                         'W-STON-STRAT-T1', 'W-SURF-STAIN-T1-JKS', 'W-SURF-GYP-JKS']

    cad_path = r"/Volumes/GoogleDrive/My Drive/Documents/Research/easternStatePenitentiary/2020_3_11/"
    walls = [r'2020-03-11 - et - DRAFT North Wall_BN.dxf',
             r'2020-03-11 - et - DRAFT East Wall_BN.dxf',
             r'2020-03-11 - et - DRAFT South Wall_BN.dxf',
             r'2020-03-11 - et - DRAFT West Wall_BN.dxf']
    directions = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}
    cad_files = [cad_path + walls[itup[0]] for itup in enumerate(walls)]

    pkl_files = glob.glob(os.path.join(cad_path, 'pkl_parallel', 'dists_*.pkl'))
    all_df = [pd.read_pickle(pf) for pf in pkl_files]
    dist_df = all_df.pop()
    dist_df.columns = [x.upper() for x in dist_df.columns]
    for df in tqdm(all_df, total=len(all_df)):
        df.columns = [x.upper() for x in df.columns]
        df.index += len(dist_df)
        dist_df = pd.merge(left=dist_df, right=df, how='outer')
    layers = sorted(dist_df.columns)
    dist_df[np.isnan(dist_df)] = np.Inf

    # read the bounding boxes from cad files
    bbox = []
    for file in cad_files:
        doc = ezdxf.readfile(file)
        # record all entities in model space
        msp = doc.modelspace()
        objs = [e for e in msp if 'boundBox' in e.dxf.layer]
        pts = [np.array([y[:2] for y in x.get_points()]) for x in objs]
        for i, pt in enumerate(pts):
            if not np.all(pt[-1] == pt[0]):
                pts[i] = np.vstack([pt, pt[0]])
        bbox.append(pts)

    # new agg code, create hierarchical categories split at dash
    h_layers = []
    for lyr in layers:
        these = []
        segments = lyr.split('-')
        for i, s in enumerate(segments):
            these.append('-'.join(segments[:i] + [s]) + '-')
        h_layers += these[:-1]
    h_layers = sorted(set(h_layers))
    # use the minimum from all matching columns
    for hl in h_layers:
        matches = [x for x in layers if hl in x]
        if len(matches) < 2:
            continue
        agg_dist = np.vstack([dist_df[m] for m in matches])
        lyr_dist = np.min(agg_dist, axis=0)
        dist_df[hl] = lyr_dist

    # for each index in wall position. find max x.y.z and put that at 0,0,0
    for wall_pos in range(4):
        wall_idx = dist_df['WALL_POSITION'] == wall_pos
        for dim in ['X', 'Y', 'Z']:  # , 'X_ORIG', 'Y_ORIG']:
            this_dim = dist_df[dim][wall_idx]
            dist_df[dim][wall_idx] -= this_dim.max()

    # create a deep copy of the original dist_df for later
    raw_dist_df = copy.deepcopy(dist_df)

    # invert the distances
    bottom_left = np.array([dist_df[dim].min() for dim in ['X', 'Y', 'Z']])
    top_right = np.array([dist_df[dim].max() for dim in ['X', 'Y', 'Z']])
    kernel_width = 0.05 * np.linalg.norm(top_right - bottom_left)
    for col in dist_df.columns:
        if col not in ['POINT_FACING', 'WALL_POSITION', 'X', 'X_ORIG', 'Y', 'Y_ORIG', 'Z', 'Z_ORIG']:
            dist_df[col] = 1.0 - np.exp(-raw_dist_df[col] / kernel_width)

    if plot_correlation:
        fig = cross_corr(dist_df, prohibited_layers)
        fig.savefig(os.path.join(cad_path, 'whole_correlation.pdf'))
        plt.close(fig)

    if plot_ground_truth:
        for certain_layer in dist_df.columns:
            if certain_layer in ['X', 'Y', 'Z', 'POINT_FACING', 'WALL_POSITION'] + prohibited_layers:
                continue
            # make predictions and plot them
            x_orig = dist_df['X_ORIG']
            y_orig = dist_df['Y_ORIG']
            Z = dist_df[certain_layer]
            #
            for w in range(4):
                fig, ax = plt.subplots(1, 1, figsize=(10, 2.5))
                w_idx = dist_df['WALL_POSITION'] == w
                plot_gt_2d(x_orig[w_idx], y_orig[w_idx], Z[w_idx],
                                 certain_layer, directions[w], bbox[w],
                                 ax=ax,
                                 cmap='Blues_r', ms=1, lw=0.5, vmin=0, vmax=1)
                fig.tight_layout()
                fig_name = '%s_%s_Wall_GT.png' % (certain_layer, directions[w])
                fig.savefig(os.path.join(cad_path, fig_name), dpi=150)
                plt.close(fig)

    predicted_layers = [l for l in layers if l not in independent_layers and l not in prohibited_layers]
    predicted_layers = predicted_layers[::-1]
    for certain_layer in tqdm(predicted_layers, total=len(predicted_layers)):
        # to get index of certain layer
        layer_order = sorted(dist_df.keys())
        new_layers = [l for l in layer_order if l not in prohibited_layers and l not in certain_layer]

        n_dist_mat = np.array([dist_df[k].values for k in new_layers])

        rand_column = np.random.rand(n_dist_mat.shape[1])
        n_dist_mat = np.vstack([n_dist_mat, rand_column])

        X = n_dist_mat.transpose()
        # X[np.isinf(X)] = 1e12  # not needed when dists are exp

        if np.all(np.unique(dist_df[certain_layer].values) == np.array([0, 1])):
            y = dist_df[certain_layer].values
        else:
            y = dist_df[certain_layer].values <= thresh

        if calculate_importance or plot_wall:
            # set aside some data for testing the model later
            test_fraction = 0.25
            test_number = int(test_fraction * X.shape[0])
            test_idx = np.random.choice(X.shape[0], test_number, replace=False)
            test_mat = np.zeros(X.shape[0], dtype=np.bool)
            test_mat[test_idx] = True
            train_mat = (test_mat == 0)
            print('working on %s' % certain_layer)
            if np.sum(y[train_mat]) == 0 or np.sum(y[train_mat] == 0) == 0:
                print('skipping %s' % certain_layer)
                continue

            # train the random forest
            rf = RandomForestClassifier(max_depth=16, random_state=42,
                                        n_estimators=64,
                                        oob_score=True)  # n_estimators=500, WANT LOWER NUMBER --> MORE GENERIC
            rf = rf.fit(X[train_mat], y[train_mat])  # train only on some part of the data

            # eval model on test data
            print('OOB score %f' % rf.oob_score_)  # if bad (< 0.7), rf cant handle this prediction
            print('score on test data %f' % rf.score(X[test_mat], y[test_mat]))  # get the score on unseen data
            # print('layer contribution to RF in descending order:')
            # for x in np.argsort(rf.feature_importances_)[::-1]:
            #     # print('%3d : %10f : %s' % (idx[x], rf.feature_importances_[x], new_layers[idx[x]]))
            #     print('%3d : %10f : %s' % (x, rf.feature_importances_[x], new_layers[x]))

        # todo: confusion matrix for predictions

        importances_df = None
        if calculate_importance:
            if importance_type == 'drop_col':  # with permutation importance
                # figure out important features
                # reduce data volume for faster evaluation
                train_fraction = 1.0
                train_number = int(train_fraction * X.shape[0])
                train_idx = np.random.choice(X.shape[0], train_number, replace=False)
                train_mat = np.zeros(X.shape[0], dtype=np.bool)
                train_mat[train_idx] = True
                importances_df = drop_col_feat_imp(rf, X[train_mat], y[train_mat], certain_layer, cad_path, new_layers)
                importances_df.to_pickle(os.path.join(cad_path, '%s_feat.pkl' % certain_layer))
            elif importance_type == 'model':  # with rf feature importance
                importances_df = pd.DataFrame(rf.feature_importances_, new_layers + ['random'])
                importances_df.to_pickle(os.path.join(cad_path, '%s_feat_model.pkl' % certain_layer))
        else:
            if importance_type == 'drop_col':  # with permutation importance
                importances_df = pd.read_pickle(os.path.join(cad_path, '%s_feat.pkl' % certain_layer))
            elif importance_type == 'model':  # with rf feature importance
                importances_df = pd.read_pickle(os.path.join(cad_path, '%s_feat_model.pkl' % certain_layer))

        if plot_importance:
            fig_title = str(certain_layer) + ' feature importance'
            fig_height = importances_df.shape[0] / 5
            ax = importances_df.plot.barh(rot=0, figsize=(8, fig_height), grid=True, title=fig_title)
            fig = ax.get_figure()
            fig.tight_layout()
            plt.show(block=False)
            rand_mag = np.abs(importances_df.values[-1])
            ax.plot(rand_mag * np.ones(2), ax.get_ylim(), 'k--')
            ax.plot(-rand_mag * np.ones(2), ax.get_ylim(), 'k--')
            plt.close(fig)
            file_name = os.path.join(cad_path, '%s_FeatureImportance.pdf' % certain_layer)
            ax.figure.savefig(file_name)

        if plot_correlation:
            fig = importance_corr(dist_df, certain_layer, importances_df, max_features=10)
            fig_name = os.path.join(cad_path, '%s_Correlation.pdf' % certain_layer)
            fig.savefig(fig_name)
            plt.close(fig)

        if plot_wall:
            # make predictions and plot them
            x_orig = dist_df['X_ORIG']
            y_orig = dist_df['Y_ORIG']
            Z = rf.predict_proba(X)[:, 1]
            #
            for w in range(4):
                w_idx = dist_df['WALL_POSITION'] == w
                fig = pred_locations_2d(x_orig[w_idx], y_orig[w_idx], y[w_idx], Z[w_idx],
                                        directions[w], certain_layer, bbox[w], cmap='RdBu_r', ms=1, lw=0.5)
                fig_name = '%s_%s_Wall.pdf' % (certain_layer, directions[w])
                fig.savefig(os.path.join(cad_path, fig_name))
                plt.close(fig)


def main():
    compute_distances(cad_files, cad_path, num_samples=1000, num_batches=1)
    run_model(thresh=10,
              plot_correlation=True,
              plot_importance=True,
              plot_wall=False,
              calculate_importance=False,
              importance_type='model')


# if __name__ == '__main__':
#     main()

def factor_analysis(dist_df, numFactors=5, prohibited_layers=[]):
    df = copy.deepcopy(dist_df)
    # regular fa
    fa = FactorAnalyzer()
    #
    df.drop([l for l in prohibited_layers if l in df.columns], axis=1, inplace=True)
    df.dropna(inplace=True)
    df[np.isinf(df)] = 1e12
    #
    df_std = np.std(df, axis=0)
    valid_std = sorted([c for c in df.columns if df_std[c] > 0])
    #
    fa.analyze(df[valid_std], numFactors, rotation='varimax')
    L = np.array(fa.loadings)
    headings = list(fa.loadings.transpose().keys())
    factor_threshold = 0.4
    factors = []
    for i, factor in enumerate(L.transpose()):
        descending = np.argsort(np.abs(factor))[::-1]
        contributions = [(np.round(factor[x], 2), headings[x]) for x in descending if
                         np.abs(factor[x]) > factor_threshold]
        factors.append(contributions)
        print('Factor %d:' % (i + 1), contributions)

    inv = False
    stacked_bars = False
    h = len(fa.loadings) / 5
    fig, ax = plt.subplots(1, 1, figsize=(6, h))
    c = np.zeros(L.shape[0])
    for i, factor in enumerate(fa.loadings.columns):
        if not stacked_bars:
            important_features = np.argsort(1 - np.abs(fa.loadings[:][factor].values))
            edgecolor = ['k' if i in important_features[:10] else 'none' for i, e in enumerate(important_features)]
            line_width = [1.5 if i in important_features[:10] else 0 for i, e in enumerate(important_features)]
            if inv:
                data = np.linalg.pinv(fa.loadings).T[:, i]
            else:
                data = fa.loadings[:][factor]
            ax.barh(fa.loadings.T.columns, data, left=c, ec=edgecolor, lw=line_width)
            c += data.max() - data.min()
        else:
            ax.barh(fa.loadings.T.columns, np.abs(fa.loadings[:][factor]), left=c)
            c += np.abs(fa.loadings[:][factor])
        # if stacked_bars:
        #     c += np.abs(fa.loadings[:][factor]).max()
        # else:
        #     c += np.abs(fa.loadings[:][factor])
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    fig.tight_layout()
    fig.savefig(os.path.join(cad_path, 'FactorAnalysis_%d_%s.pdf' % (numFactors, stacked_bars)))
    plt.close(fig)

    if plot_wall:
        # make predictions and plot them
        x_orig = dist_df['X_ORIG']
        y_orig = dist_df['Y_ORIG']
        for i, factor in enumerate(fa.loadings.columns):
            Z = np.zeros(len(df))
            inv_loadings = np.linalg.pinv(fa.loadings)
            # inv_loadings = np.array(fa.loadings).T
            for j, col in enumerate(fa.loadings.T.columns):
                Z += inv_loadings[i, j] * (df[col] - np.mean(df[col])) / np.std(df[col])
            #
            mag = np.median(np.abs(Z)) * 5
            #
            fig, ax = plt.subplots(4, 1, figsize=(10, 10))
            for w in range(4):
                w_idx = df['WALL_POSITION'] == w
                plot_gt_2d(x_orig[w_idx], y_orig[w_idx], Z[w_idx],
                           factor, directions[w], bbox[w],
                           cmap='RdBu_r', ms=1, lw=0.5, ax=ax[w], )
                # vmin=-mag, vmax=mag)
            fig_name = '%stotal_%s.png' % (numFactors, factor)
            fig.savefig(os.path.join(cad_path, fig_name), dpi=150)
            plt.close(fig)

    factor_list = []
    weight_list = []
    for i, factor in enumerate(factors):
        factor_list.append([])
        weight_list.append([])
        for j, condition in enumerate(factor):
            factor_list[i].append(condition[1])
            weight_list[i].append(condition[0])
    #
    # for m, entry in enumerate(factor_list):
    #     n = int(np.ceil(len(entry) / 3))
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
