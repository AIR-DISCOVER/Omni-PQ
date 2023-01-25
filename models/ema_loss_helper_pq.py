import os, sys
import torch
import torch.nn as nn

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness
GT_VOTE_FACTOR = 3 # number of GT votes per point
QUAD_CLS_WEIGHTS = [0.4,0.6] 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from nn_distance import nn_distance, huber_loss
sys.path.append(BASE_DIR)
from utils.losses import smoothl1_loss, SigmoidFocalClassificationLoss
import time
from box_util import get_3d_box


def compute_quad_score_loss_ema(end_points, num_layer=6):
    # Associate proposal and GT objects by point-to-point distances
    prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_layer-1)]
    quad_score_loss_sum = 0.0
   
    for prefix in prefixes:
    
        
        gt_center = end_points['gt_quad_centers'][:, :, 0:3]  # B, K2, 3
        aggregated_vote_xyz = end_points['aggregated_sample_xyz']
        B = gt_center.shape[0]
        K = aggregated_vote_xyz.shape[1]
        K2 = gt_center.shape[1]
        dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2
        num_gt_quads = end_points["num_gt_quads"] #Bx1

        euclidean_dist1 = torch.sqrt(dist1+1e-6)
        quad_label = torch.zeros((B,K), dtype=torch.long).cuda()
        quad_mask = torch.zeros((B,K)).cuda()
        quad_label[euclidean_dist1<NEAR_THRESHOLD] = 1
        # end_points['last_quad_label_pseudo_gt'] stores 0/1 variables recording whether or not a prediction is quad
        # if end_points['last_quad_label_pseudo_gt'][b][ind1[b][k]] is 1, then quad_label[b][k] == 1, vice versa
        
        for b in range(end_points['last_quad_label_pseudo_gt'].shape[0]):
            end_points['last_quad_label_pseudo_gt'][b]
            quad_label[b] = end_points['last_quad_label_pseudo_gt'][b][ind1[b]]
        quad_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
        quad_mask[euclidean_dist1>FAR_THRESHOLD] = 1

        # Set assignment
        quad_assignment = ind1
        quad_assignment[quad_label==0] = K2 - 1  # set background points to the last gt bbox
        
        end_points[f'{prefix}quad_label'] = quad_label
        end_points[f'{prefix}quad_mask'] = quad_mask
        end_points[f'{prefix}quad_assignment'] = quad_assignment

        # Compute quad scores loss
        quad_scores = end_points[f'{prefix}quad_scores']
        criterion = nn.CrossEntropyLoss(torch.Tensor(QUAD_CLS_WEIGHTS).cuda(), reduction='none')
        quad_scores_loss = criterion(quad_scores.transpose(2,1), quad_label)  # Calc binary classification loss per "quad" to decide if it is a quad
        quad_scores_loss = torch.sum(quad_scores_loss * quad_mask)/(torch.sum(quad_mask)+1e-6)

        end_points[f'{prefix}quad_scores_loss'] = quad_scores_loss
        quad_score_loss_sum += quad_scores_loss

    return quad_score_loss_sum, end_points