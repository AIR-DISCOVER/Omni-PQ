from turtle import distance
import torch
from torch import tensor
import torch.nn as nn
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from nn_distance import nn_distance, huber_loss
sys.path.append(BASE_DIR)
from utils.losses import smoothl1_loss, SigmoidFocalClassificationLoss
import time
from box_util import get_3d_box

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness
GT_VOTE_FACTOR = 3 # number of GT votes per point
QUAD_CLS_WEIGHTS = [0.4,0.6] 


def compute_vote_loss(end_points):
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += end_points['seed_xyz'].repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    # If votes in bounding box, then minimize its distance to the ground truth
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss


def compute_objectness_loss(end_points,num_layer = 6):
    """ Compute objectness loss for the proposals."""

    # Associate proposal and GT objects by point-to-point distances
    prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_layer-1)]
    objectness_loss_sum = 0.0
   
    for prefix in prefixes:
        # Associate proposal and GT objects
        gt_center = end_points['center_label'][:, :, 0:3]  # B, K2, 3
        aggregated_vote_xyz = end_points['aggregated_vote_xyz']
        B = gt_center.shape[0]
        K = aggregated_vote_xyz.shape[1]
        K2 = gt_center.shape[1]
        dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2
        num_gt_boxes = end_points["num_gt_boxes"] #Bx1

        euclidean_dist1 = torch.sqrt(dist1+1e-6)
        objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
        objectness_mask = torch.zeros((B,K)).cuda()
        objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
        objectness_label[ind1>=num_gt_boxes] = 0
        objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
        objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

        # Set assignment
        object_assignment = ind1
        object_assignment[objectness_label==0] = K2 - 1  # set background points to the last gt bbox
        
        end_points[f'{prefix}objectness_label'] = objectness_label
        end_points[f'{prefix}objectness_mask'] = objectness_mask
        end_points[f'{prefix}object_assignment'] = object_assignment

        # Compute objectness loss
        objectness_scores = end_points[f'{prefix}objectness_scores']
        criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
        objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
        objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

        end_points[f'{prefix}objectness_loss'] = objectness_loss
        objectness_loss_sum += objectness_loss

    return objectness_loss_sum, end_points


def compute_box_and_sem_cls_loss(end_points, config,num_layer = 6):
    """ Compute 3D bounding box and semantic classification loss. """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr
    batch_size = end_points['center_label'].shape[0]
    box_loss_sum = 0.0
    sem_cls_loss_sum = 0.0
    
    prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_layer-1)]
    box_loss_sum = 0.0
    sem_cls_loss_sum = 0.0
    for prefix in prefixes:
        object_assignment = end_points[f'{prefix}object_assignment']
        batch_size = object_assignment.shape[0]
        # Compute center loss
        pred_center = end_points[f'{prefix}center']
        gt_center = end_points['center_label'][:, :, 0:3]

        objectness_label = end_points[f'{prefix}objectness_label'].float()
        object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
        assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
        center_loss = smoothl1_loss(assigned_gt_center - pred_center, delta=1.0)  # (B,K)
        center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)

        # Compute heading loss
        heading_class_label = torch.gather(end_points['heading_class_label'], 1,
                                           object_assignment)  # select (B,K) from (B,K2)
        criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
        heading_class_loss = criterion_heading_class(end_points[f'{prefix}heading_scores'].transpose(2, 1),
                                                     heading_class_label)  # (B,K)
        heading_class_loss = torch.sum(heading_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        heading_residual_label = torch.gather(end_points['heading_residual_label'], 1,
                                              object_assignment)  # select (B,K) from (B,K2)
        heading_residual_normalized_label = heading_residual_label / (np.pi / num_heading_bin)

        # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
        heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1],
                                                       num_heading_bin).zero_()
        heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1),
                                       1)  # src==1 so it's *one-hot* (B,K,num_heading_bin)
        heading_residual_normalized_error = torch.sum(
            end_points[f'{prefix}heading_residuals_normalized'] * heading_label_one_hot,
            -1) - heading_residual_normalized_label

        
        heading_residual_normalized_loss = 1.0 * smoothl1_loss(heading_residual_normalized_error,
                                                                        delta=1.0)  # (B,K)
        heading_residual_normalized_loss = torch.sum(
            heading_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        # Compute size loss
        size_class_label = torch.gather(end_points['size_class_label'], 1,
                                        object_assignment)  # select (B,K) from (B,K2)
        criterion_size_class = nn.CrossEntropyLoss(reduction='none')
        size_class_loss = criterion_size_class(end_points[f'{prefix}size_scores'].transpose(2, 1),
                                               size_class_label)  # (B,K)
        size_class_loss = torch.sum(size_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        size_residual_label = torch.gather(
            end_points['size_residual_label'], 1,
            object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)

        size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
        size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1),
                                    1)  # src==1 so it's *one-hot* (B,K,num_size_cluster)
        size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3)  # (B,K,num_size_cluster,3)
        predicted_size_residual_normalized = torch.sum(end_points[f'{prefix}size_residuals_normalized'] * size_label_one_hot_tiled, 2)  # (B,K,3)

        mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(
            0)  # (1,1,num_size_cluster,3)
        mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2)  # (B,K,3)
        size_residual_label_normalized = size_residual_label / mean_size_label  # (B,K,3)

        size_residual_normalized_error = predicted_size_residual_normalized - size_residual_label_normalized

        size_residual_normalized_loss = 1.0 * smoothl1_loss(size_residual_normalized_error,
                                                                    delta=1.0)  # (B,K,3) -> (B,K)
        size_residual_normalized_loss = torch.sum(size_residual_normalized_loss * objectness_label.unsqueeze(2)) / (
                torch.sum(objectness_label) + 1e-6)

        # 3.4 Semantic cls loss
        sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
        criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
        sem_cls_loss = criterion_sem_cls(end_points[f'{prefix}sem_cls_scores'].transpose(2, 1), sem_cls_label)  # (B,K)
        sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        end_points[f'{prefix}center_loss'] = center_loss
        end_points[f'{prefix}heading_cls_loss'] = heading_class_loss
        end_points[f'{prefix}heading_reg_loss'] = heading_residual_normalized_loss
        end_points[f'{prefix}size_cls_loss'] = size_class_loss
        end_points[f'{prefix}size_reg_loss'] = size_residual_normalized_loss
        box_loss = center_loss + 0.1 * heading_class_loss + heading_residual_normalized_loss + 0.1 * size_class_loss + size_residual_normalized_loss
        end_points[f'{prefix}box_loss'] = box_loss
        end_points[f'{prefix}sem_cls_loss'] = sem_cls_loss

        box_loss_sum += box_loss
        sem_cls_loss_sum += sem_cls_loss
    return box_loss_sum, sem_cls_loss_sum, end_points



def compute_quad_score_loss(end_points,num_layer = 6):
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
        quad_label[ind1>=num_gt_quads] = 0
        quad_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
        quad_mask[euclidean_dist1>FAR_THRESHOLD] = 1
        
        # Set assignment
        quad_assignment = ind1
        quad_assignment[quad_label==0] = K2 - 1  # set background points to the last gt bbox

        # # TODO: Set the label of quads with mismatched sizes to 0
        # filter_cond = (euclidean_dist1<NEAR_THRESHOLD) & ~(ind1>=num_gt_quads)
        # pred_center = end_points[f'{prefix}quad_center']
        # gt_center = end_points['gt_quad_centers'][:, :, 0:3]
        # quad_assignment_expand = quad_assignment.unsqueeze(2).repeat(1, 1, 3)
        # assigned_gt_center = torch.gather(gt_center, 1, quad_assignment_expand)  # (B, K, 3) from (B, K2, 3)
        # center_loss = assigned_gt_center - pred_center  # (B,K)
        # center_delta = torch.norm(center_loss, dim=2)
        
        # pred_size = end_points[f'{prefix}quad_size']
        # gt_size = torch.gather(end_points['gt_quad_sizes'], 1, 
        #                     quad_assignment.unsqueeze(2).repeat(1, 1, 2)) # (B, K, 3) from (B, K2, 3)
        # size_loss = (pred_size - gt_size) / 2
        # size_delta = torch.norm(size_loss, dim=2)
        # total_delta = center_delta + size_delta
        
        # quad_label[total_delta > 0.4] = 0

        
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


def compute_quad_loss(end_points, config,num_layer = 6):
    """ Compute 3D bounding box and semantic classification loss. """

    quad_center_loss_sum = 0.0
    quad_vector_loss_sum = 0.0
    quad_size_loss_sum = 0.0
    #quad_direction_loss_sum = 0.0
    prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_layer-1)]
    for prefix in prefixes:
        quad_assignment = end_points[f'{prefix}quad_assignment']
        # To decide if a prediction is considered as a quad or not
        quad_label = end_points[f'{prefix}quad_label'].float()
        B = quad_assignment.shape[0]
        K = quad_assignment.shape[1]

        # Compute center loss
        pred_center = end_points[f'{prefix}quad_center']
        gt_center = end_points['gt_quad_centers'][:, :, 0:3]
        quad_assignment_expand = quad_assignment.unsqueeze(2).repeat(1, 1, 3)
        assigned_gt_center = torch.gather(gt_center, 1, quad_assignment_expand)  # (B, K, 3) from (B, K2, 3)
        center_loss = smoothl1_loss(assigned_gt_center - pred_center, delta=1.0)  # (B,K)
        center_loss = torch.sum(center_loss * quad_label.unsqueeze(2)) / (torch.sum(quad_label) + 1e-6)
        end_points[f'{prefix}quad_center_loss'] = center_loss
        quad_center_loss_sum += center_loss

        # Compute normal vector loss
        pred_vector = end_points[f'{prefix}normal_vector'] #B,K,3
        gt_vector = torch.gather(end_points['gt_normal_vectors'],1,quad_assignment_expand)

        cos_similar =  torch.cosine_similarity(pred_vector,gt_vector,dim=2)
        vector_loss = torch.ones((B,K), dtype=torch.float32).cuda()-cos_similar  # (B,K)
        #vector_loss = smoothl1_loss(pred_vector - gt_vector, delta=1.0)  # (B,K)
        vector_loss = torch.sum(vector_loss * quad_label) / (torch.sum(quad_label) + 1e-6)
        end_points[f'{prefix}normal_vector_loss'] = vector_loss
        quad_vector_loss_sum += vector_loss

        # Compute size loss
        pred_size = end_points[f'{prefix}quad_size']
        gt_size = torch.gather(end_points['gt_quad_sizes'], 1, 
                            quad_assignment.unsqueeze(2).repeat(1, 1, 2)) # (B, K, 3) from (B, K2, 3)
        size_loss = smoothl1_loss(pred_size - gt_size,delta=1.0)
        size_loss = torch.sum((size_loss * quad_label.unsqueeze(2)) / (
                        torch.sum(quad_label) + 1e-6))
        end_points[f'{prefix}quad_size_loss'] = size_loss
        quad_size_loss_sum += size_loss

    return quad_center_loss_sum,quad_vector_loss_sum,quad_size_loss_sum, end_points


def get_2d_box(box_size,center):
    l = box_size[:,:,0]
    w = box_size[:,:,1]
    h = box_size[:,:,2]
    
    bsize = center.shape[0]
    num_proposal = center.shape[1]
    
    pred_corners_2d = torch.zeros((bsize, num_proposal, 4, 2)).cuda()

    pred_corners_2d[:,:,0,0] = pred_corners_2d[:,:,1,0] = l/2
    pred_corners_2d[:,:,2,0] = pred_corners_2d[:,:,3,0] =-l/2
    pred_corners_2d[:,:,0,1] = pred_corners_2d[:,:,2,1] = w/2
    pred_corners_2d[:,:,1,1] = pred_corners_2d[:,:,3,1] = -w/2

    for i in range(4):
        pred_corners_2d[:,:,i,0] = pred_corners_2d[:,:,i,0] + center[:,:,0]
        pred_corners_2d[:,:,i,1] = pred_corners_2d[:,:,i,1] + center[:,:,1]

    return pred_corners_2d
    
def projection2d(point,center,normal_vector,size):
    P = point.shape[0]
    a = normal_vector[0]
    b = normal_vector[1]
    d = -(a*center[0]+b*center[1])

    k = -(a*point[:,0]+b*point[:,1]+d) # k = -delta, k < 0 indicates the point is inside the quads

    x = point[:,0]+a*k
    y = point[:,1]+b*k

    t = torch.cat((x.reshape(P,1),y.reshape(P,1)),dim=-1)
    w = torch.norm(t-center[0:2],dim=1)

    point_mask = torch.zeros(P).cuda()    
    collision = torch.zeros(P).cuda()    

    point_mask[w<size[0]] = 1    
    quad = torch.cat((a.view([1]),b.view([1])))
    delta = point.matmul(quad)+d
    pc_loss = torch.relu(-delta)*point_mask
    collision[pc_loss>1e-4]=1
    return pc_loss.sum(),collision.sum()

def not_door_or_window(sem_class):
    if (sem_class == 5) or (sem_class == 6) or (sem_class==8) or (sem_class== 11):
        return False
    return True

def compute_physical_constraints_loss(end_points,config):
    pc_loss = 0.0
    #start_time = time.time()
    mean_size_arr = torch.from_numpy(config.mean_size_arr).cuda()
    prefixes = ['last_'] #+ [f'{i}head_' for i in range(5)]

    for prefix in prefixes:
        pred_center = end_points[f'{prefix}center']  # B,num_proposal,3
        pred_size_class = torch.argmax(end_points[f'{prefix}size_scores'], -1)  # B,num_proposal
        pred_size_residual = torch.gather(end_points[f'{prefix}size_residuals'], 2,pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,3))  # B,num_proposal,1,3
        pred_size_residual.squeeze_(2)
        objectness_label = end_points[f'{prefix}objectness_label'].float()
        object_assignment = end_points[f'{prefix}object_assignment']
        object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
        sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment)
        num_proposal = pred_center.shape[1]
        bsize = pred_center.shape[0]

        # use gt boxes
        # gt_center = end_points['center_label'][:, :, 0:3]
        # gt_size = end_points['size_gts']
        # assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  
        # assigned_gt_size = torch.gather(gt_size, 1, object_assignment_expand)  
        # pred_corners_2d = get_2d_box(assigned_gt_size,assigned_gt_center)
        
        #use predicted boxes
        box_size = mean_size_arr[pred_size_class, :] + pred_size_residual
        pred_corners_2d = get_2d_box(box_size,pred_center)
        
        pred_quad_center = end_points[f'{prefix}quad_center']
        pred_normal_vector = end_points[f'{prefix}normal_vector']
        pred_quad_size = end_points[f'{prefix}quad_size']
        quad_label = end_points[f'{prefix}quad_label']
        num_quad = pred_quad_center.shape[1]

        collisions = 0

        for i in range(bsize):
            num_box = 0
            batch_pred_corners_2d = []
            for j in range(0,num_proposal):
                if objectness_label[i,j] and not_door_or_window(sem_cls_label[i,j]):
                    num_box = num_box + 1
                    batch_pred_corners_2d.append(pred_corners_2d[i,j])
            if len(batch_pred_corners_2d) == 0:
                continue
            corner_point = torch.cat(tuple(batch_pred_corners_2d), 0)
            for k in range(num_quad):
                if quad_label[i,k]:
                    loss,collision = projection2d(corner_point,pred_quad_center[i,k],pred_normal_vector[i,k],pred_quad_size[i,k])
                    pc_loss = pc_loss + loss/num_box
                    collisions = collisions + collision

    return pc_loss,collisions

def get_loss(end_points, config,query_points_obj_topk=5,pc_loss=True,num_layer = 6):
    """ Loss functions

    Args:
        end_points: dict
            {   
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """


    if  'vote_xyz' in end_points.keys():
        vote_loss = compute_vote_loss(end_points)
    else:
        vote_loss = 0.0
    
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss_sum, end_points =  compute_objectness_loss(end_points)
    end_points['objectness_loss'] = objectness_loss_sum
    
    # Box loss and sem cls loss
    box_loss_sum, sem_cls_loss_sum, end_points = compute_box_and_sem_cls_loss(end_points, config)
    end_points['box_loss'] = box_loss_sum
    end_points['sem_cls_loss_sum'] = sem_cls_loss_sum

    # quadness loss
    quad_score_loss_sum, end_points = compute_quad_score_loss(end_points)
    end_points['quad_score_loss_sum'] = quad_score_loss_sum
    
    # quad loss
    quad_center_loss_sum,quad_vector_loss_sum, quad_size_loss_sum, end_points = compute_quad_loss(end_points, config)
    end_points['quad_center_loss_sum'] = quad_center_loss_sum
    end_points['quad_vector_loss_sum'] = quad_vector_loss_sum
    end_points['quad_size_loss_sum'] = quad_size_loss_sum

    quad_loss_sum = quad_center_loss_sum + quad_vector_loss_sum + quad_size_loss_sum
    end_points['quad_loss_sum'] = quad_loss_sum

    #pc loss
    if pc_loss:
        pc_loss,collisions = compute_physical_constraints_loss(end_points,config)
    else:
        pc_loss = 0.0
        collisions = 0

    end_points['physical_constraints_loss'] = pc_loss
    end_points['collisions'] = collisions

   
    object_loss = box_loss_sum + 0.1*sem_cls_loss_sum + 0.5*objectness_loss_sum
    quad_loss = quad_loss_sum+ 0.5*quad_score_loss_sum
    # Final loss function
    loss = pc_loss + vote_loss + 1.0/(num_layer+1) *(0.9*object_loss + 0.1*quad_loss)
    
    loss *= 10
    end_points['loss'] = loss



    return loss, end_points

def get_loss_distance(end_points, config, query_points_obj_topk=5, pc_loss=False, num_layer=6):
    from models.utils.distance_util import distance_loss
    loss1 = distance_loss(end_points, config, query_points_obj_topk, pc_loss, num_layer)
    
    lambda_distance = 0.1
    
    if end_points['use_gt'].any():
        loss2, end_points = get_loss(end_points, config, query_points_obj_topk, pc_loss, num_layer)
        end_points['loss'] = lambda_distance * loss1 + loss2
    else:
        end_points['loss'] = loss1 * lambda_distance
    
    return end_points['loss'], end_points

def get_loss_teacher(end_points, ema_end_points, config, query_points_obj_topk=5, pc_loss=False, num_layer=6):

    def rotate_xyz(xyz, augment_flip_YZ_XZ, rot_angle):
        """Rotate a point cloud back.

        Args:
            xyz (torch.Tensor or np.ndarray): Point clouds
            augment_flip_YZ_XZ (_type_): _description_
            rot_angle (_type_): _description_

        Returns:
            _type_: _description_
        """
        import pc_util, copy
        rot_mat = pc_util.rotz(-rot_angle)
        
        if isinstance(xyz, torch.Tensor):
            xyz = xyz.cpu().numpy()
        
        former_shape = list(xyz.shape)
        new_shape = copy.deepcopy(former_shape)
        new_shape.append(1)
        
        pc = np.matmul(rot_mat, xyz.reshape(new_shape)).reshape(former_shape)
        
        if augment_flip_YZ_XZ[1]:
            # Flip XZ
            pc[:, 1] = -1 * pc[:, 1]  
            
        
        if augment_flip_YZ_XZ[0]:
            # Flip YZ
            pc[:, 0] = -1 * pc[:, 0]
        
        return pc
        
        
    batch_size = end_points['point_clouds'].shape[0]
    ema_end_points['last_quad_center_rotate_back'] = ema_end_points['last_quad_center'].cpu().numpy()
    for b in range(batch_size):
        # First, let's rotate back the prediction of quad!
        augment_flip_YZ_XZ = end_points['augment_flip'][b].cpu().numpy()
        rot_angle = end_points['augment_rot'][b].cpu().numpy()
        ema_end_points['last_quad_center_rotate_back'][b] = rotate_xyz(
            ema_end_points['last_quad_center'][b], augment_flip_YZ_XZ, rot_angle
        )
        
        
    ### Quad Score Loss: use compute_quad_score_loss
    """
    Requires:
        gt_quad_centers
        aggregated_sample_xyz
        num_gt_quads
        {prefix}quad_scores
    """
    
    quad_score_end_points = {}
    
    quad_score_end_points['aggregated_sample_xyz'] = end_points['aggregated_sample_xyz']
    # Use the same prediction of ema_end_points for ground truth
    quad_score_end_points['gt_quad_centers'] = torch.tensor(ema_end_points['last_quad_center_rotate_back']).cuda()
    
    s = end_points['last_quad_label'].shape
    quad_score_end_points['num_gt_quads'] = torch.tensor(np.array(s[1]).repeat(s[0]).repeat(s[1]).reshape(s)).cuda()
    # quad_score_end_points['num_gt_quads'] =\
    #     np.repeat(end_points['last_quad_label'].sum(axis=1).cpu().numpy(), s[1]).reshape(s[0], -1)

    prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_layer-1)]
    for prefix in prefixes:
        quad_score_end_points[f'{prefix}quad_scores'] = end_points[f'{prefix}quad_scores'] 
        quad_score_end_points[f'{prefix}quad_label_pseudo_gt'] = end_points[f'{prefix}quad_label']
        quad_score_end_points[f'{prefix}quad_center'] = end_points[f'{prefix}quad_center']
        quad_score_end_points[f'{prefix}quad_size'] = end_points[f'{prefix}quad_size']
        quad_score_end_points[f'{prefix}normal_vector'] = end_points[f'{prefix}normal_vector']


    import IPython
    IPython.embed(header="get_loss_teacher")

    from models.ema_loss_helper_pq import compute_quad_score_loss_ema
    quad_score_loss_sum, quad_score_end_points = compute_quad_score_loss_ema(quad_score_end_points)
    quad_score_end_points['quad_score_loss_sum'] = quad_score_loss_sum
    print(quad_score_loss_sum)
    
    
    ### Quad Loss
    """
    Requires:
        {prefix}quad_assignment
        {prefix}quad_label
        
        {prefix}quad_center  => Smooth L1, center loss
        gt_quad_centers
        
        {prefix}normal_vector => Cosine similarity, normal_vector_loss
        gt_normal_vectors
        
        {prefix}quad_size => Smooth L1, quad_size_loss
        gt_quad_sizes
    """
    quad_score_end_points['gt_quad_sizes'] = ema_end_points['last_quad_size'].detach()
    # quad_score_end_points['gt_normal_vectors']
    augment_normal = ema_end_points['last_normal_vector']
    augment_normal = augment_normal.cpu().numpy().astype(np.float64)
    # augment_normal = nn.functional.normalize(augment_normal, dim=2)
    _ = augment_normal.shape
    any_direction = torch.tensor([0, 0, 1], dtype=torch.float64).repeat(_[:2]).reshape(_).cuda()
    any_direction = np.array([[0., 0., 1.]], dtype=np.float64).repeat(_[1], axis=0)[np.newaxis, :, :].repeat(_[0], axis=0)

    
    # v1 and v2 are orthogonal vectors vertical to augment_normal, which lies on the normal plane
    v1 = np.cross(augment_normal, any_direction)
    v1 = v1 / np.linalg.norm(v1, axis=2)[:, :, np.newaxis]
    v2 = np.cross(augment_normal, v1)
    v2 = v2 / np.linalg.norm(v2, axis=2)[:, :, np.newaxis]
    
    pt1_back = ema_end_points['last_quad_center'].cpu().numpy().astype(np.float64) + v1
    pt2_back = ema_end_points['last_quad_center'].cpu().numpy().astype(np.float64) + v2
    
    # Rotate pt1 and pt2 back
    for b in range(batch_size):
        # First, let's rotate back the prediction of quad!
        augment_flip_YZ_XZ = end_points['augment_flip'][b].cpu().numpy()
        rot_angle = end_points['augment_rot'][b].cpu().numpy()
        
        pt1_back[b] = rotate_xyz(pt1_back[b], augment_flip_YZ_XZ, rot_angle)
        pt2_back[b] = rotate_xyz(pt2_back[b], augment_flip_YZ_XZ, rot_angle)

    v1_back = pt1_back - quad_score_end_points['gt_quad_centers'].cpu().numpy().astype(np.float64)
    v2_back = pt2_back - quad_score_end_points['gt_quad_centers'].cpu().numpy().astype(np.float64)
    v1_back = v1_back / np.linalg.norm(v1_back, axis=2)[:, :, np.newaxis]
    v2_back = v2_back / np.linalg.norm(v2_back, axis=2)[:, :, np.newaxis]
    normal_back = np.cross(v1_back, v2_back)
    normal_back = normal_back / np.linalg.norm(normal_back, axis=2)[:, :, np.newaxis]
    
    quad_score_end_points['gt_normal_vectors'] = torch.tensor(normal_back).float().cuda()
    
    quad_center_loss_sum,quad_vector_loss_sum, quad_size_loss_sum, quad_score_end_points =\
        compute_quad_loss(quad_score_end_points, config)
    quad_score_end_points['quad_center_loss_sum'] = quad_center_loss_sum
    quad_score_end_points['quad_vector_loss_sum'] = quad_vector_loss_sum
    quad_score_end_points['quad_size_loss_sum'] = quad_size_loss_sum
    quad_loss_sum = quad_center_loss_sum + quad_vector_loss_sum + quad_size_loss_sum
    

    quad_loss = quad_loss_sum + 2.5 * quad_score_loss_sum
    
    return quad_loss


def get_loss_mean_teacher(end_points, ema_end_points, config, query_points_obj_topk=5, pc_loss=False, num_layer=6):
    
    gt_loss, gt_end_points = get_loss(end_points, config, query_points_obj_topk=query_points_obj_topk, pc_loss=pc_loss, num_layer=num_layer)
    teacher_loss = get_loss_teacher(end_points, ema_end_points, config, query_points_obj_topk=5, pc_loss=pc_loss, num_layer=6)
    # from models.utils.distance_util import distance_loss
    # distance_l = distance_loss(end_points, config, query_points_obj_topk=5, pc_loss=pc_loss, num_layer=6)
    
    print(end_points['use_gt'])
    if not end_points['use_gt'].any():
        lambda_gt_loss = 0.0  # Mask it up
    else:
        lambda_gt_loss = 1.0
    
    lambda_teacher = 0.0
    # lambda_distance = 0.1
    new_loss = lambda_teacher * teacher_loss + lambda_gt_loss * gt_loss
    
    end_points = gt_end_points
    end_points['gt_loss'] = gt_loss
    end_points['teacher_loss'] = teacher_loss
    # end_points['distance_loss'] = distance_l
    end_points['loss'] = new_loss
    
    return new_loss, end_points
