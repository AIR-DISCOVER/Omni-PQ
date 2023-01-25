import numpy as np
import torch
import torch.nn as nn
from models.dump_helper_quad import dump_results_quad, dump_single_quad
from models.dump_helper import dump_results, dump_pc, dump_pc_colored
from models.utils.distance_util import Palette
from models.utils.losses import smoothl1_loss

FAR_THRESHOLD = 0.30
NEAR_THRESHOLD = 0.04
QUAD_CLS_WEIGHTS = [0.4, 0.6]
GM_CLIP = 0.85

def viz_distance_histogram(distance, save_name):
    if isinstance(distance, torch.Tensor):
        distance = distance.detach().cpu().numpy()
    
    from matplotlib import pyplot as plt
    plt.cla()
    plt.hist(distance, bins=1000, color='g', alpha=0.50)
    plt.savefig(save_name)


ABLATION_FILTERING = False
ABLATION_EPS_D = 0.2

def quad_point_mixture_metric(quad_center, normal_vector, quad_size, quad_score, downsampled_pc, downsampled_normal, prefix="last_", config=None, **kwargs):

    quad_size[0] /= 1.5

    downsampled_K = downsampled_pc.shape[0]
    p = torch.softmax(quad_score, dim=-1)[1]

    normal_vectors_predicted = normal_vector  # (3, )
    normal_vectors_predicted = normal_vectors_predicted[:2] / torch.norm(normal_vectors_predicted[:2], dim=0).detach()  # (2, )
    normal_vectors_predicted = torch.concat([normal_vectors_predicted, torch.zeros(size=(1, )).cuda()])

    # Component A : Cosine similarity
    normal_vectors_pc = downsampled_normal
    normal_vectors_pc = normal_vectors_pc / torch.norm(normal_vectors_pc, dim=1)[:, None].clamp_(min=1e-5)
    distance_cosine = 1. -torch.matmul(normal_vectors_predicted.view(1, 1, 3),
                                       normal_vectors_pc.view(downsampled_K, 3, 1))[..., 0, 0].abs()
    
    # Component B : vertical distance
    point_offset_to_predicted_quad_center = downsampled_pc - quad_center
    vertical_distance_matrix = torch.matmul(point_offset_to_predicted_quad_center.reshape(downsampled_K, 1, 3), 
            normal_vectors_predicted.detach().view(1, 3, 1))[..., 0, 0]
    vertical_distance = vertical_distance_matrix.abs()
    
    # Component C : Size Penalty
    z_dir = torch.tensor([0., 0., 1.]).cuda()
    x_dir = torch.cross(z_dir, normal_vectors_predicted)
    
    x_dis = torch.matmul(point_offset_to_predicted_quad_center.reshape(downsampled_K, 1, 3), x_dir.reshape(1, 3, 1))[..., 0, 0].abs()
    z_dis = torch.matmul(point_offset_to_predicted_quad_center.reshape(downsampled_K, 1, 3), z_dir.reshape(1, 3, 1))[..., 0, 0].abs()
    size_distance_A = torch.norm((2 * torch.stack([x_dis, z_dis], dim=1) - quad_size).clamp(min=0.), dim=-1)
    size_distance_B = torch.norm((quad_size * 0.25 - torch.stack([x_dis, z_dis], dim=1)).clamp(min=0.), dim=-1)
    
    total_distance = 2.5 * distance_cosine + 0.2 * (size_distance_A ** 2) + 0.5 * vertical_distance
    
    # Gamma Mixture Fitting: Point selection
    if not ABLATION_FILTERING:
        old_settings = np.seterr(all='ignore')
        from fit import fit_gamma
        mask_label = fit_gamma(total_distance.detach().cpu().numpy(), a1=2, b1=20, a2=3, b2=1, weight=0.1, step=25,\
                        # save=None)
                        save=None if kwargs['save_name'] is None else "./dump/fig/" + kwargs['save_name'], quiet=True)
        keep_mask = ~torch.tensor(mask_label)
        np.seterr(**old_settings)
    else:
        keep_mask = (vertical_distance < ABLATION_EPS_D) & (vertical_distance < 0.1)
    

    # Return distance metric
    kept_pts = downsampled_pc[keep_mask, ...]
    kept_normals = downsampled_normal[keep_mask, ...]
    
    if kept_pts.shape[0] < 300:  # Ignore the case of unstable
        return torch.tensor(0.).cuda(), torch.tensor(0.).cuda(), torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
    else:
        
        estimated_normal = kept_normals.mean(dim=0)[:2]
        estimated_normal = torch.concat([estimated_normal, torch.zeros((1,)).cuda()], dim=0)
        estimated_normal = estimated_normal / torch.norm(estimated_normal)
        
        # metric_cosine
        metric_normal = 1.0 - torch.cosine_similarity(
            estimated_normal[None, ...], normal_vectors_predicted[None, ...]
        ).abs().item()
        
        # metric vertical
        metric_vertical_matrix = vertical_distance[keep_mask]
        metric_vertical = (metric_vertical_matrix * (metric_vertical_matrix < torch.quantile(metric_vertical_matrix, GM_CLIP))).mean()
        
        # metric size: penalty quads way too small or way too large
        kept_pt_mean = torch.mean(kept_pts, dim=0)
        point_offset_to_kept_pt_mean = downsampled_pc - kept_pt_mean
        x_dis = torch.matmul(point_offset_to_kept_pt_mean.reshape(downsampled_K, 1, 3),\
            x_dir.reshape(1, 3, 1))[..., 0, 0].abs()
        z_dis = torch.matmul(point_offset_to_kept_pt_mean.reshape(downsampled_K, 1, 3),\
            z_dir.reshape(1, 3, 1))[..., 0, 0].abs()
        
        metric_x_dis = x_dis[keep_mask]
        metric_z_dis = z_dis[keep_mask]
        
        thresholds = np.linspace(0.85, 1.00, num=3)
        x_candidates, z_candidates = [], []
        for thres in thresholds:
            x_candidates.append(torch.quantile(metric_x_dis, thres) / thres)
            z_candidates.append(torch.quantile(metric_z_dis, thres) / thres)
        pseudo_x_dis, pseudo_z_dis = torch.stack(x_candidates).mean(), \
                                     torch.stack(z_candidates).mean()

        metric_size = smoothl1_loss(quad_size[0] - 2 * torch.tensor([pseudo_x_dis]).cuda()).sum()
        metric_size += 0. * smoothl1_loss(quad_size[1] - 2 * torch.tensor([pseudo_z_dis]).cuda()).sum()
        metric_size += smoothl1_loss(torch.mean(kept_pts[..., :], dim=0) - quad_center[:]).sum()
        
        # metric score
        score_criterion = torch.nn.CrossEntropyLoss()
        if (metric_vertical < 0.05 and metric_normal < 0.02 and metric_size < 0.10):
            metric_score = score_criterion(quad_score[None, ...], torch.tensor(1).cuda()[None, ...])
        elif (metric_vertical > 0.3 or metric_normal > 0.05 or metric_size > 0.35):
            metric_score = score_criterion(quad_score[None, ...], torch.tensor(0).cuda()[None, ...])
        else:
            metric_score = torch.tensor(0.0).cuda()
        
        return metric_normal, metric_vertical, metric_size, metric_score
    

def gamma_mixture_guide_criterion(end_points, DATASET_CONFIG, config, **kwargs):
    
    mixture_metric_normal, mixture_metric_vertical, mixture_metric_size, mixture_metric_score = \
        torch.tensor(0.).cuda(), torch.tensor(0.).cuda(), torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
    
    prefix = "last_"
    point_clouds = end_points['point_clouds']
    point_normals = end_points['vertex_normals']
    
    quad_scores = end_points[f"{prefix}quad_scores"]
    quad_centers = end_points[f"{prefix}quad_center"]
    normal_vectors = end_points[f"{prefix}normal_vector"]
    quad_sizes = end_points[f"{prefix}quad_size"]
    
    quad_masks = torch.softmax(quad_scores, dim=-1)[..., 1] > 0.1
    batch_size, max_quad_num = quad_masks.shape[0], quad_masks.shape[1]
    num_pc = point_clouds.shape[1]
    
    
    for b in range(batch_size):
        # For each scene, randomly pick 1 quad to optimize
        quad_mask = quad_masks[b]
        ind_ = torch.where(quad_mask)[0]
        
        skip = False
        
        if ind_.shape[0] == 0:
            ind = torch.randint(0, max_quad_num, size=(1,)).cuda()[0]
            skip = True
        elif ind_.shape[0] == 1:
            ind = ind_[0]
        elif ind_.shape[0] >= 2:
            import random
            ind = random.choice(ind_)
        
        # Debugging
        # for ind in ind_:

        if not skip:
            # Filter out points belonging to this quad
            quad_score = quad_scores[b, ind]
            quad_center = quad_centers[b][ind]
            normal_vector = normal_vectors[b][ind]
            quad_size = quad_sizes[b][ind]
            
            # Calculate "distance" from down-sampled point-clouds to this quad
            K = 10000
            downsample_inds = torch.randint(0, num_pc, (K, ))
            downsampled_normal = point_normals[b, downsample_inds, ...]
            downsampled_pc = point_clouds[b, downsample_inds, ...]
            
            metric_normal, metric_vertical, metric_size, metric_score = quad_point_mixture_metric(quad_center, normal_vector, quad_size, quad_score,\
                                        downsampled_pc, downsampled_normal,\
                                        prefix=prefix, config=config, DATASET_CONFIG=DATASET_CONFIG,\
                                        save_name=None, **kwargs)
            mixture_metric_normal += metric_normal
            mixture_metric_vertical += metric_vertical
            mixture_metric_size += metric_size
            mixture_metric_score += metric_score



    return mixture_metric_normal / batch_size, mixture_metric_vertical / batch_size, mixture_metric_size / batch_size, mixture_metric_score / batch_size