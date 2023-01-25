import numpy as np
import torch
import os, time

from utils.logger import setup_logger
from scannet.model_util_scannet import ScannetDatasetConfig
from models.dump_helper_quad import dump_results_quad

class PaletteColor:
    def __init__(self, primary_color, secondary_color):
        self.primary_color = np.array(primary_color)
        self.secondary_color = np.array(secondary_color)
    
    def __call__(self, distance, out):
        if out: 
            # return self.secondary_color
            return np.array((76, 76, 76))
        
        distance = float(abs(distance))
        distance = 1.0 if distance > 1.0 else distance
        return self.primary_color * (1.0 - distance) + (self.secondary_color + self.primary_color) * (distance) / 2


class Palette:
    def __init__(self):
        self.idx = 0
        self.raw_colors = [
            ((58, 232, 27), (177, 221, 169)),
            ((240, 247, 0), (248, 250, 173)),
            ((13, 42, 250), (204, 209, 245)),
            ((250, 7, 250), (254, 217, 254)),
            ((173, 0, 254), (232, 205, 244)),
            ((255, 0, 0), (255, 208, 198)),
        ]
        self.colors = [PaletteColor(color[0], color[1]) for color in self.raw_colors]

    def __call__(self, idx, distance, out):
        idx = idx % 6
        # assert 0 <= idx < 6, "You need to increase your color number if you want to "
        return self.colors[int(idx)](distance, out)


def viz_distance(end_points):
    """
    Visualize distance loss.
    Would be saved in `./statistics/s{scene_idx}-{time_str}/`
    """

    # Prepare output directory
    scene_idx = end_points['scan_name'][0][len("scene"):][:-len("_01")]
    time_str = time.strftime("%Y%m%d%H%M%S")
    save_dir = os.path.join(f"./statistics/s{scene_idx}-{time_str}/")
    os.makedirs(save_dir, exist_ok=True)

    # Output predicted bounding boxes
    DATASET_CONFIG = ScannetDatasetConfig()
    dump_results_quad(end_points, os.path.join(save_dir), DATASET_CONFIG)
    
    for x in [x for x in os.listdir(save_dir) if scene_idx not in x]:
        os.remove(os.path.join(save_dir, x))
    


    prefixes = ["last_"]
    
    # Layout mask
    point_clouds = end_points['point_clouds'][0]
    semantic_labels = end_points['semantic_labels'][0]
    mask = semantic_labels == 1  # 40000
    mask = torch.bitwise_or(mask, semantic_labels == 8)
    mask = torch.bitwise_or(mask, semantic_labels == 9)
    not_layout_mask = [not x for x in mask]

    for prefix in prefixes:

        with open(os.path.join(save_dir, "point_cloud.txt"), 'w') as f:
            
            # 1. Points not fallen into layout categories
            not_layout_points = point_clouds[not_layout_mask, :]  # 22552, 3
            for point_xyz in not_layout_points:
                print(f"{point_xyz[0]} {point_xyz[1]} {point_xyz[2]} 0.20 0.20 0.20", file=f)

            # 2. Points fallen into layout categories
            palette = Palette()
            layout_points = point_clouds[mask, :]  # 17448, 3

            distance = end_points[f"{prefix}distance"][mask]  # 17448
            pred_idx = end_points[f"{prefix}distance_idx_array"][mask]  # 17448
            
            # Filter points
            kept_idx = end_points[f"{prefix}keep_label"]  # 17448

            # 2.1 Points filtered out
            filtered_out_idx = [not x for x in kept_idx]
            for idx, point_xyz in enumerate(layout_points[filtered_out_idx]):   # (0, 3), which means no points being filtered out
                color_idx = pred_idx[filtered_out_idx][idx]
                color = palette(color_idx, 1.0, True)  # Anyway it is filtered out
                print(f"{point_xyz[0]} {point_xyz[1]} {point_xyz[2]} {float(color[0]/255)} {float(color[1]/255)} {float(color[2]/255)}", file=f)
            
            # 2.2 Points left
            pred_idx_2 = pred_idx[kept_idx]
            distance_idx_2 = distance[kept_idx]

            for idx, point_xyz in enumerate(layout_points[kept_idx]):  # (17448, 3), all points left
                color_idx = pred_idx_2[idx]
                distance_point = distance_idx_2[idx]
                color = palette(color_idx, distance_point, False)
                print(f"{point_xyz[0]} {point_xyz[1]} {point_xyz[2]} {float(color[0]/255)} {float(color[1]/255)} {float(color[2]/255)}", file=f)


def calc_distance_vertically(_pc_scene, predicted_quads):

    # TODO: Use Tensor instead of np.ndarray

    pc_scene = _pc_scene.cuda()
    pc_center = torch.mean(pc_scene, dim=0)  # To find the inner side of the point clouds
    distance = 10.0 * torch.ones((pc_scene.shape[0],), dtype=torch.float).cuda()

    idx_array = torch.ones((pc_scene.shape[0],), dtype=torch.float).cuda()

    # Calculate distances
    for predict_idx, _predicted_quad in enumerate(predicted_quads):
        if isinstance(_predicted_quad, np.ndarray):
            _predicted_quad = torch.tensor(_predicted_quad)

        predicted_quad = _predicted_quad.cuda()
        # TODO: convert params predicted_quad to torch.Tensor before and after nms to form a complete compute map
        quad_center = torch.mean(predicted_quad, dim=0)
        predicted_quad_norm = torch.cross(predicted_quad[1] - predicted_quad[0], predicted_quad[2] - predicted_quad[0])
        predicted_quad_norm = predicted_quad_norm / torch.norm(predicted_quad_norm)

        if torch.dot(pc_center - quad_center, predicted_quad_norm) > 0:
            predicted_quad_norm = -predicted_quad_norm  # Make inner distance < 0 and outsider > 0

        vertical_distance = (pc_scene - quad_center) @ predicted_quad_norm
        vertical_distance = vertical_distance.float()
        # vertical_distance = torch.bmm((pc_scene - quad_center).view(num_points, 1, 3),
        #                               predicted_quad_norm.repeat(num_points, 1).view(num_points, 3, 1)) \
        #     .view(num_points, )

        # Use Dynamic programming to find the "nearest" quad with minimum absolute error :)
        mask = torch.abs(vertical_distance) < torch.abs(distance)
        distance[mask] = vertical_distance[mask]
        idx_array[mask] = predict_idx

    return distance, idx_array


def calc_distance_from_center(_pc_scene, predicted_quads, lambda_l=0):
    pc_scene = _pc_scene.cuda()
    pc_center = torch.mean(pc_scene, dim=0)
    distance = 10.0 * torch.ones((pc_scene.shape[0],), dtype=torch.float).cuda()

    # Calculate distances per quad
    for _predicted_quad in predicted_quads:
        predicted_quad = torch.tensor(_predicted_quad).cuda()
        # TODO: convert params predicted_quad to torch.Tensor before and after nms to form a complete compute map
        quad_center = torch.mean(predicted_quad, dim=0)
        predicted_quad_norm = torch.cross(predicted_quad[1] - predicted_quad[0], predicted_quad[2] - predicted_quad[0])
        predicted_quad_norm = predicted_quad_norm / torch.norm(predicted_quad_norm)

        if torch.dot(pc_center - quad_center, predicted_quad_norm) > 0:
            predicted_quad_norm = -predicted_quad_norm  # Make inner distance < 0 and outsider > 0


        vertical_distance = (pc_scene - quad_center) @ predicted_quad_norm

        # Calc parallel loss: penalty those points whose projection lying outside the quads
        # Norm: (x, y ,z) -> (-y, x, z)
        parallel_norm1 = predicted_quad[1] - predicted_quad[0]
        parallel_norm1 /= torch.norm(parallel_norm1)
        parallel_norm2 = torch.cross(predicted_quad_norm, parallel_norm1)
        parallel_norm2 /= torch.norm(parallel_norm2)
        limit1_p = torch.dot(predicted_quad[1] - quad_center, parallel_norm1)
        limit2_p = torch.dot(predicted_quad[2] - quad_center, parallel_norm2)

        parallel_distance1 = torch.relu(torch.abs((pc_scene - quad_center) @ parallel_norm1) - torch.abs(limit1_p))
        parallel_distance2 = torch.relu(torch.abs((pc_scene - quad_center) @ parallel_norm2) - torch.abs(limit2_p))
        cond = parallel_distance1 > parallel_distance2
        parallel_distance = torch.where(cond, parallel_distance1, parallel_distance2)

        new_distance = lambda_l * torch.abs(vertical_distance) + (1-lambda_l) * parallel_distance
        # new_distance = parallel_distance
        distance[torch.abs(new_distance) < torch.abs(distance)] = new_distance[torch.abs(new_distance) < torch.abs(distance)]

    return distance

def calc_distance_quad_center_penalty():
    pass


def distance_loss_spectral_clustering(end_points, config, query_points_obj_topk, pc_loss, num_layer):

    points, semantic_labels = end_points['point_clouds'], end_points['semantic_labels']
    batch_size = points.shape[0]
    CONFIG_DICT = {'remove_empty_box': False, 'use_3d_nms': True,
                   'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
                   'per_class_proposal': True, 'conf_thresh': 0.0,'quad_thresh':0.5,
                   'dataset_config': None} 
        
    # Prepare layout categories
    mask = semantic_labels == 1
    mask = torch.bitwise_or(mask, semantic_labels == 8)
    mask = torch.bitwise_or(mask, semantic_labels == 9)

    # prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_layer-1)]
    prefixes = ['last_']
    
    for prefix in prefixes:
        
        from models.ap_helper_pq import parse_quad_predictions
        batch_pred_map_cls, pred_mask, batch_pred_corners_list = parse_quad_predictions(end_points, CONFIG_DICT, prefix)

        if prefix == "last_":
            end_points['pred_quad_mask'] = pred_mask

        for b in range(batch_size):
            
            import warnings
            warnings.filterwarnings('ignore')

            # Retrieve quads
            point_cloud, semantic_label, vertex_normal = points[b], semantic_labels[b], end_points['vertex_normals'][b]
            pred_corner = end_points[f'{prefix}batch_pred_corners_list_tensor'][b]

            # Prepare layout points
            layout_point_mask = mask[b]
            layout_point_cloud = point_cloud[layout_point_mask, :]
            layout_pt_cnt = layout_point_cloud.shape[0]
            
            layout_pc_center = layout_point_cloud.mean(axis=0)

            import random
            SELECTION = 1000
            selected_choices = random.sample(range(layout_point_cloud.shape[0]), SELECTION)
            selected_pts = layout_point_cloud[selected_choices, :].cpu().numpy()
            
            # Euclid: mean: 3.94, std: 1.95, max: 7.64, min: 0.00
            euclid = np.sqrt(np.sum((selected_pts.reshape(SELECTION, 1, 3) - selected_pts.reshape(1, SELECTION, 3)) ** 2, axis=2))

            # Normal similarity: calc and statistic analysis
            import open3d as o3d
            param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=5)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(layout_point_cloud.cpu().numpy())
            o3d.geometry.estimate_normals(pcd,
                search_param=param)
            normals = np.asarray(pcd.normals)
            
            reverse_mask = ((layout_point_cloud - layout_pc_center).cpu().numpy().reshape(layout_pt_cnt, 1, 3) \
                    @ normals.reshape(layout_pt_cnt, 3, 1)).reshape(layout_pt_cnt) < 0
            
            normals[reverse_mask] = -normals[reverse_mask]
            
            
            selected_normals = normals[selected_choices, :]
            # np.any(np.abs(normals[:, :2]) > 0.8, axis=1).sum()
            
            # selected_normals = vertex_normal[layout_point_mask, :][selected_choices].cpu().numpy()
            s1 = selected_normals.reshape(SELECTION, 1, 1, 3)
            s2 = selected_normals.reshape(1, SELECTION, 3 ,1)
            
            # mean: 0.92, std: 0.61, max: 2.00, min: 0.00
            cosine_distance = 1 - (s1@s2).reshape(SELECTION, SELECTION) + 1e-5
            # cosine_similarity = cosine_similarity / np.abs(np.max(cosine_similarity))
            # cosine_distance = np.arccos(cosine_similarity) / np.pi
            
            # D distance
            # Mean: 2.38, max 34.94, std 3.93
            d = - selected_normals.reshape(SELECTION, 1, 3) @ selected_pts.reshape(SELECTION, 3, 1)
            d_distance = np.abs(d.reshape(SELECTION, 1) - d.reshape(1, SELECTION)) ** 2

            # Distance
            _lambda = (0.1, 1.0, 0.)
            distance = _lambda[0] * euclid + _lambda[1] * cosine_distance + _lambda[2] * d_distance
            
            # Spectral Clustering
            std = distance.std()
            matrix_Z = np.exp( -distance ** 2 / (2 * (std**2)) )
            matrix_D = np.diag(np.sum(matrix_Z, axis=1) **(-0.5))
            matrix_Z_tilter = matrix_D @ matrix_Z @ matrix_D
            eigen_value, eigen_vector = np.linalg.eig(matrix_Z_tilter)
            
            if end_points['use_gt'].any():
                num_gt_quads = end_points['num_gt_quads'][b][0].item()
                print(f"gt num quad {num_gt_quads}  threshold {eigen_value[num_gt_quads-1]} {eigen_value[num_gt_quads]} avg {(eigen_value[num_gt_quads-1]+eigen_value[num_gt_quads])/2}")
                print(eigen_value[:6])
            
            THRESHOLD_K = 0.1
            K = int((eigen_value > THRESHOLD_K).sum())
            
            from sklearn import manifold, cluster
            label_pred = cluster.spectral_clustering(n_clusters=K, affinity=matrix_Z_tilter)

            palette = Palette()
            with open(f"stats0707/{end_points['scan_name'][b]}_{K}.txt", 'w+') as file:
                for k in range(SELECTION):
                    color = palette(label_pred[k], 0.0, False)
                    file.write(f"{selected_pts[k][0]} {selected_pts[k][1]} {selected_pts[k][2]} {color[0]/255} {color[1]/255} {color[2]/255} {selected_normals[k][0]} {selected_normals[k][1]} {selected_normals[k][2]} 1.0\n")
            
            # distance_embed = manifold.TSNE(metric="precomputed", learning_rate=200, n_iter=1500).fit_transform(distance)
            
            # from matplotlib import pyplot as plt
            # plt.cla()
            # plt.scatter(distance_embed[:, 0], distance_embed[:, 1], c='g')
            # plt.savefig("distance.png")
    return 0


def distance_loss_gamma_mixture(end_points, config, query_points_obj_topk, pc_loss, num_layer):

    distance_loss = 0.0

    points, semantic_labels = end_points['point_clouds'], end_points['semantic_labels']
    batch_size = points.shape[0]
    CONFIG_DICT = {'remove_empty_box': False, 'use_3d_nms': True,
                   'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
                   'per_class_proposal': True, 'conf_thresh': 0.0,'quad_thresh':0.5,
                   'dataset_config': None} 
        
    # Prepare layout categories
    mask = semantic_labels == 1
    mask = torch.bitwise_or(mask, semantic_labels == 8)
    mask = torch.bitwise_or(mask, semantic_labels == 9)

    # prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_layer-1)]
    prefixes = ['last_']

    total_fit = 0

    for prefix in prefixes:
        from models.ap_helper_pq import parse_quad_predictions
        batch_pred_map_cls, pred_mask, batch_pred_corners_list = parse_quad_predictions(end_points, CONFIG_DICT, prefix)

        if prefix == "last_":
            end_points['pred_quad_mask'] = pred_mask

        for b in range(batch_size):
            
            import warnings
            warnings.filterwarnings('ignore')

            # Retrieve quads
            point_cloud, semantic_label, vertex_normal = points[b], semantic_labels[b], end_points['vertex_normals'][b]
            pred_corner = end_points[f'{prefix}batch_pred_corners_list_tensor'][b]

            # Prepare layout points
            layout_point_mask = mask[b]
            layout_point_cloud = point_cloud[layout_point_mask, :]

            # Calculate Distances
            distance, idx_array = calc_distance_vertically(point_cloud, pred_corner)
            distance_layout = distance[layout_point_mask]

            if b == 0:
                end_points[f'{prefix}distance'] = distance  # Save distance for visualization
                end_points[f'{prefix}distance_idx_array'] = idx_array
            
            # For the pleasure of eval process
            keep_label = [True for x in range(distance_layout.shape[0])]
            if b == 0:
                end_points[f'{prefix}keep_label'] = keep_label  # Save keep label

            # Filter out points
            import fit
            # from fit import fit_gamma
            # filter_out_label = fit_gamma(torch.abs(distance_layout).detach().cpu().numpy())
            runner = fit.FitRunner([(fit.GammaDistribution, (2, 40)), (fit.GammaDistribution, (10,20))], np.abs(distance_layout.detach().cpu().numpy()))
            runner.fit(step=20, quiet=True, visualize=False, save='test.png', opt=True)

            init_a = (runner.dist_a.params[0] - 1) / (runner.dist_a.params[1])
            init_b = (runner.dist_b.params[0] - 1) / (runner.dist_b.params[1])

            mask_second_a = runner.judge2(distance_layout.detach().cpu().numpy(), (init_a + init_b) / 2)  # If true, keep
            
            keep_label = [x for x in mask_second_a]
            
            distance_left = distance_layout[keep_label]

            if distance_left.shape[0] == 0:
                print("Warning! No point left after filtering!")
                distance_loss += 0.0
                # distance_loss += torch.mean(torch.abs(distance_layout))  # Use L1 Loss
            else:
                # Calculate distance loss and accumulate
                distance_loss += torch.mean(torch.abs(distance_left))  # Use L1 Loss
                total_fit += 1

    # assert(total_fit != 0)
    if total_fit > 0:
        distance_loss /= total_fit

    lambda_distance = 1.
    return lambda_distance * distance_loss

def distance_loss(end_points, config, query_points_obj_topk, pc_loss, num_layer):
    return distance_loss_gamma_mixture(end_points, config, query_points_obj_topk, pc_loss, num_layer)
    # return distance_loss_spectral_clustering(end_points, config, query_points_obj_topk, pc_loss, num_layer)