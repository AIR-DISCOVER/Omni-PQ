import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from nn_distance import nn_distance, huber_loss

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness
EMA_CLIP = 0.85

ABLATION_CONFIDENT = False

def compute_center_consistency_loss(end_points, ema_end_points, prefix="last_"):
    
    center = end_points[f'{prefix}center'] #(B, num_proposal, 3)
    ema_center = ema_end_points[f'{prefix}center'] #(B, num_proposal, 3)
    flip_x_axis = end_points['flip_x_axis'] #(B,)
    flip_y_axis = end_points['flip_y_axis'] #(B,)
    rot_mat = end_points['rot_mat'] #(B,3,3)
    scale_ratio = end_points['scale'] #(B,1,3)

    # align ema_center with center based on the input augmentation steps
    inds_to_flip_x_axis = torch.nonzero(flip_x_axis).squeeze(1)
    ema_center[inds_to_flip_x_axis, :, 0] = -ema_center[inds_to_flip_x_axis, :, 0]

    inds_to_flip_y_axis = torch.nonzero(flip_y_axis).squeeze(1)
    ema_center[inds_to_flip_y_axis, :, 1] = -ema_center[inds_to_flip_y_axis, :, 1]

    ema_center = torch.bmm(ema_center, rot_mat.transpose(1,2)) #(B, num_proposal, 3)

    ema_center = ema_center * scale_ratio.reshape((scale_ratio.shape[0], 1, 1))
    end_points[f'{prefix}ema_center'] = ema_center
    
    scores = F.softmax(end_points[f'{prefix}objectness_scores'], dim=2)[..., 1]

    dist1, ind1, dist2, ind2 = nn_distance(center, ema_center)  #ind1 (B, num_proposal): find the ema_center index closest to center
    dist1_mask = torch.stack([score[ind] for score, ind in zip(scores, ind1)], dim=0)
    dist2_mask = scores
    
    end_points[f'{prefix}ema_assignment'] = ind2
    end_points[f'{prefix}ema_assignment_confidence'] = dist2_mask

    if not ABLATION_CONFIDENT:
        dist = dist1 * dist1_mask + dist2 * dist2_mask
    else:
        dist = dist1 + dist2
    
    eps = torch.quantile(dist, EMA_CLIP)
    dist_ = (dist<eps) * dist
    return torch.mean(dist_), ind2

def compute_center_consistency_loss_quad(end_points, ema_end_points, prefix="last_"):
    center = end_points[f'{prefix}quad_center']
    ema_center = ema_end_points[f'{prefix}quad_center']
    flip_x_axis = end_points['flip_x_axis'] #(B,)
    flip_y_axis = end_points['flip_y_axis'] #(B,)
    rot_mat = end_points['rot_mat'] #(B,3,3)
    scale_ratio = end_points['scale'] #(B,1,3)
    
    # align ema_center with center based on the input augmentation steps
    inds_to_flip_x_axis = torch.nonzero(flip_x_axis).squeeze(1)
    ema_center[inds_to_flip_x_axis, :, 0] = -ema_center[inds_to_flip_x_axis, :, 0]

    inds_to_flip_y_axis = torch.nonzero(flip_y_axis).squeeze(1)
    ema_center[inds_to_flip_y_axis, :, 1] = -ema_center[inds_to_flip_y_axis, :, 1]

    ema_center = torch.bmm(ema_center, rot_mat.transpose(1,2)) #(B, num_proposal, 3)
    ema_center = ema_center * scale_ratio.reshape((scale_ratio.shape[0], 1, 1))
    end_points[f'{prefix}ema_center_quad'] = ema_center
    
    scores = F.softmax(end_points[f'{prefix}quad_scores'], dim=2)[..., 1]
    
    dist1, ind1, dist2, ind2 = nn_distance(center, ema_center)  #ind1 (B, num_proposal): ema_center index closest to center
    dist1_mask = torch.stack([score[ind] for score, ind in zip(scores, ind1)], dim=0)
    dist2_mask = scores
    
    end_points[f'{prefix}ema_assignment_quad'] = ind2
    end_points[f'{prefix}ema_assignment_quad_confidence'] = dist2_mask

    if not ABLATION_CONFIDENT:
        dist = dist1 * dist1_mask + dist2 * dist2_mask
    else:
        dist = dist1 + dist2
    
    eps = torch.quantile(dist, EMA_CLIP)
    dist_ = (dist<eps) * dist
    return torch.mean(dist_), ind2
    
    

def compute_class_consistency_loss(end_points, ema_end_points, map_ind, prefix="last_"):
    cls_scores = end_points[f'{prefix}sem_cls_scores'] #(B, num_proposal, num_class)
    ema_cls_scores = ema_end_points[f'{prefix}sem_cls_scores'] #(B, num_proposal, num_class)

    cls_log_prob = F.log_softmax(cls_scores, dim=2) #(B, num_proposal, num_class)
    # cls_log_prob = F.softmax(cls_scores, dim=2)
    ema_cls_prob = F.softmax(ema_cls_scores, dim=2) #(B, num_proposal, num_class)

    cls_log_prob_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(cls_log_prob, map_ind)])

    class_consistency_loss = F.kl_div(cls_log_prob_aligned, ema_cls_prob, reduction='mean')
    # class_consistency_loss = F.mse_loss(cls_log_prob_aligned, ema_cls_prob)
    return class_consistency_loss*2

# Unused
def compute_class_consistency_loss_quad(end_points, ema_end_points, map_ind, prefix="last_"):
    cls_scores = end_points[f'{prefix}quad_scores'] #(B, num_proposal, num_class)
    ema_cls_scores = ema_end_points[f'{prefix}quad_scores'] #(B, num_proposal, num_class)
    
    cls_log_prob = F.log_softmax(cls_scores, dim=2) #(B, num_proposal, num_class)
    # cls_log_prob = F.softmax(cls_scores, dim=2)
    ema_cls_prob = F.softmax(ema_cls_scores, dim=2) #(B, num_proposal, num_class)
    
    cls_log_prob_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(cls_log_prob, map_ind)])

    class_consistency_loss = F.kl_div(cls_log_prob_aligned, ema_cls_prob, reduction='batchmean')
    # class_consistency_loss = F.mse_loss(cls_log_prob_aligned, ema_cls_prob)

    return class_consistency_loss*2


def compute_size_consistency_loss(end_points, ema_end_points, map_ind, config, prefix="last_"):
    mean_size_arr = config.mean_size_arr
    mean_size_arr = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda() #(num_size_cluster,3)
    B, K = map_ind.shape

    scale_ratio = end_points['scale'] #(B,1,3)
    size_class = torch.argmax(end_points[f'{prefix}size_scores'], -1) # B,num_proposal
    size_residual = torch.gather(end_points[f'{prefix}size_residuals'], 2, size_class.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,3)) # B,num_proposal,1,3
    size_residual.squeeze_(2)

    ema_size_class = torch.argmax(ema_end_points[f'{prefix}size_scores'], -1) # B,num_proposal
    ema_size_residual = torch.gather(ema_end_points[f'{prefix}size_residuals'], 2, ema_size_class.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,3)) # B,num_proposal,1,3
    ema_size_residual.squeeze_(2)

    size_base = torch.index_select(mean_size_arr, 0, size_class.view(-1))
    size_base = size_base.view(B,K,3)
    size = size_base + size_residual

    ema_size_base = torch.index_select(mean_size_arr, 0, ema_size_class.view(-1))
    ema_size_base = ema_size_base.view(B,K,3)
    ema_size = ema_size_base + ema_size_residual
    ema_size = ema_size * scale_ratio.reshape((scale_ratio.shape[0], 1, 1))

    size_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(size, map_ind)])

    # size_consistency_loss = F.mse_loss(size_aligned, ema_size)
    dist = torch.sum((size_aligned - ema_size) ** 2, dim=2)
    
    if not ABLATION_CONFIDENT:
        dist = dist * end_points[f'{prefix}ema_assignment_confidence']
    else:
        dist = dist
    
    eps = torch.quantile(dist, EMA_CLIP)
    dist_ = (dist<eps) * dist
    return torch.mean(dist_)


def compute_size_consistency_loss_quad(end_points, ema_end_points, map_ind, config, prefix="last_"):
    B, K = map_ind.shape
    # Normals
    normal = end_points[f'{prefix}normal_vector']
    ema_normal = ema_end_points[f'{prefix}normal_vector']
    normal_aligned = torch.cat([
        torch.index_select(a, 0, i).unsqueeze(0)
        for a, i in zip(normal, map_ind)
    ])
    normal_consistency_dist = 1. - F.cosine_similarity(
        normal_aligned[..., :2], ema_normal[..., :2], dim=2
    ).abs()
    if not ABLATION_CONFIDENT:
        normal_consistency_dist = normal_consistency_dist * end_points[f'{prefix}ema_assignment_quad_confidence']
    normal_consistency_eps = torch.quantile(normal_consistency_dist, EMA_CLIP)
    normal_consistency_dist_ = (normal_consistency_dist < normal_consistency_eps) * normal_consistency_dist
    
    
    # Sizes
    size = end_points[f'{prefix}quad_size']
    ema_size= ema_end_points[f'{prefix}quad_size']
    size_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(size, map_ind)])
    # size_consistency_loss = F.mse_loss(size_aligned, ema_size)
    size_consistency_dist = torch.sum((size_aligned - ema_size) ** 2, dim=2)
    if not ABLATION_CONFIDENT:
        size_consistency_dist = size_consistency_dist * end_points[f'{prefix}ema_assignment_quad_confidence']
    size_consistency_eps = torch.quantile(size_consistency_dist, EMA_CLIP)
    size_consistency_dist_ = (size_consistency_dist<size_consistency_eps) * size_consistency_dist
    
    return torch.mean(normal_consistency_dist_), torch.mean(size_consistency_dist_)
    
    

def get_consistency_loss(end_points, ema_end_points, config):
    """
    Args:
        end_points: dict
            {
                center, size_scores, size_residuals_normalized, sem_cls_scores,
                flip_x_axis, flip_y_axis, rot_mat
            }
        ema_end_points: dict
            {
                center, size_scores, size_residuals_normalized, sem_cls_scores,
            }
    Returns:
        consistency_loss: pytorch scalar tensor
        end_points: dict
    """
    
    center_consistency_loss_sum = torch.tensor(0.).cuda()
    class_consistency_loss_sum = torch.tensor(0.).cuda()
    size_consistency_loss_sum = torch.tensor(0.).cuda()
    consistency_loss_sum = torch.tensor(0.).cuda()
    quad_center_consistency_loss_sum = torch.tensor(0.).cuda()
    quad_class_consistency_loss_sum = torch.tensor(0.).cuda()
    quad_normal_consistency_loss_sum = torch.tensor(0.).cuda()
    quad_size_consistency_loss_sum = torch.tensor(0.).cuda()
    quad_consistency_loss_sum = torch.tensor(0.).cuda()
    
    prefixes = ['last_', 'proposal_'] + [f'{i}head_' for i in range(5)]
    # prefixes = ['last_']
    
    for prefix in prefixes:
    
        # Objects
        center_consistency_loss, map_ind = compute_center_consistency_loss(end_points, ema_end_points, prefix=prefix)
        class_consistency_loss = compute_class_consistency_loss(end_points, ema_end_points, map_ind, prefix=prefix)
        size_consistency_loss = compute_size_consistency_loss(end_points, ema_end_points, map_ind, config, prefix=prefix)

        consistency_loss =  0.5 * center_consistency_loss + 1.0 * class_consistency_loss + 0.05 * size_consistency_loss
        
        center_consistency_loss_sum += center_consistency_loss
        class_consistency_loss_sum += class_consistency_loss
        size_consistency_loss_sum += size_consistency_loss
        consistency_loss_sum += consistency_loss
        
        # Quads
        quad_center_consistency_loss, quad_map_ind = compute_center_consistency_loss_quad(end_points, ema_end_points, prefix=prefix)
        quad_class_consistency_loss = compute_class_consistency_loss_quad(end_points, ema_end_points, quad_map_ind, prefix=prefix)
        quad_normal_consistency_loss, quad_size_consistency_loss = compute_size_consistency_loss_quad(end_points, ema_end_points, quad_map_ind, config, prefix=prefix)
        quad_consistency_loss = 0.5 * quad_center_consistency_loss + 0. * quad_class_consistency_loss + 1.0 * quad_normal_consistency_loss + 0.05 * quad_size_consistency_loss

        quad_center_consistency_loss_sum += quad_center_consistency_loss
        quad_class_consistency_loss_sum += quad_class_consistency_loss
        quad_normal_consistency_loss_sum += quad_normal_consistency_loss
        quad_size_consistency_loss_sum += quad_size_consistency_loss
        quad_consistency_loss_sum += quad_consistency_loss


    end_points['center_consistency_loss'] = center_consistency_loss_sum / len(prefixes)
    end_points['class_consistency_loss'] = class_consistency_loss_sum / len(prefixes)
    end_points['size_consistency_loss'] = size_consistency_loss_sum / len(prefixes)
    end_points['consistency_loss'] = consistency_loss_sum / len(prefixes)
    
    end_points['quad_center_consistency_loss_sum'] = quad_center_consistency_loss_sum / len(prefixes)
    end_points['quad_class_consistency_loss_sum'] = quad_class_consistency_loss_sum / len(prefixes)
    end_points['quad_normal_consistency_loss_sum'] = quad_normal_consistency_loss_sum / len(prefixes)
    end_points['quad_size_consistency_loss_sum'] = quad_size_consistency_loss_sum / len(prefixes)
    end_points['quad_consistency_loss_sum'] = quad_consistency_loss_sum / len(prefixes)
    

    return (consistency_loss_sum / len(prefixes) + quad_consistency_loss_sum / len(prefixes)), end_points
