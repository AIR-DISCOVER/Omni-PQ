""" Helper functions and class to calculate Average Precisions for 3D object detection and 3D layout estimation.
"""
import os
import sys
import numpy as np
from numpy.core.defchararray import center
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.function_base import vectorize
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from eval_det import eval_det_cls, eval_det_multiprocessing
from eval_det import get_iou_obb
from nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls
from box_util import get_3d_box, get_3d_box_tensor

from models.utils.ap_util import extract_pc_in_box3d

MAX_NUM_QUAD = 32
LENGTH = 0.5
QUAD_THRES = 0.995


def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[..., 1] *= -1
    return pc2

def flip_axis_to_camera_tensor(pc):
    ''' 
    Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array

    Tensor version.
    '''

    pc2 = torch.clone(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # cam X, Y, Z = depth X, -Z, Y
    pc2[..., 1] *= -1
    return pc2


def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
    pc2[..., 2] *= -1
    return pc2

def filp_axis_to_depth_tensor(pc):
    pc2 = torch.clone(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
    pc2[..., 2] *= -1
    return pc2


def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


def sigmoid(x):
    ''' Numpy function for sigmoid'''
    s = 1 / (1 + np.exp(-x))
    return s

def parse_predictions(end_points, config_dict, prefix=""):
    """ Parse predictions to OBB parameters and suppress overlapping boxes
    
    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """

    pred_center = end_points[f'{prefix}center']  # B,num_proposal,3
    pred_heading_class = torch.argmax(end_points[f'{prefix}heading_scores'], -1)  # B,num_proposal
    pred_heading_residual = torch.gather(end_points[f'{prefix}heading_residuals'], 2,
                                         pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_size_class = torch.argmax(end_points[f'{prefix}size_scores'], -1)  # B,num_proposal
    pred_size_residual = torch.gather(end_points[f'{prefix}size_residuals'], 2,
                                      pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                         3))  # B,num_proposal,1,3
    pred_size_residual.squeeze_(2)
    pred_sem_cls = torch.argmax(end_points[f'{prefix}sem_cls_scores'], -1)  # B,num_proposal
    sem_cls_probs = softmax(end_points[f'{prefix}sem_cls_scores'].detach().cpu().numpy())  # B,num_proposal,10

    num_proposal = pred_center.shape[1]
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    # pred_size_check = end_points[f'{prefix}pred_size']  # B,num_proposal,3
    # pred_bbox_check = end_points[f'{prefix}bbox_check']  # B,num_proposal,3

    bsize = pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict['dataset_config'].class2angle( \
                pred_heading_class[i, j].detach().cpu().numpy(), pred_heading_residual[i, j].detach().cpu().numpy())
            box_size = config_dict['dataset_config'].class2size( \
                int(pred_size_class[i, j].detach().cpu().numpy()), pred_size_residual[i, j].detach().cpu().numpy())
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, pred_center_upright_camera[i, j, :])
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

    K = pred_center.shape[1]  # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))

    if config_dict['remove_empty_box']:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = end_points['point_clouds'].cpu().numpy()[:, :, 0:3]  # B,N,3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
                box3d = flip_axis_to_depth(box3d)
                pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i, j] = 0
        # -------------------------------------

    obj_logits = end_points[f'{prefix}objectness_scores'].detach().cpu().numpy()
    obj_prob = sigmoid(obj_logits)[:, :, 1]  # (B,K)
    #print(obj_prob)
    if not config_dict['use_3d_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_2d_with_prob[j, 2] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_2d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_2d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            pick = nms_2d_faster(boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                 config_dict['nms_iou'], config_dict['use_old_type_nms'])
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]

            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and (not config_dict['cls_nms']):
        # ----------
        #  NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 7))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                 config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and config_dict['cls_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[i, j]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                         config_dict['nms_iou'], config_dict['use_old_type_nms'])
            # assert (len(pick) > 0)
            if len(pick) > 0:
                pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------
    
    batch_pred_map_cls = []  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        if config_dict['per_class_proposal']:
            cur_list = []
            for ii in range(config_dict['dataset_config'].num_class):
                cur_list += [(ii, pred_corners_3d_upright_camera[i, j], sem_cls_probs[i, j, ii] * obj_prob[i, j]) \
                             for j in range(pred_center.shape[1]) if
                             pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']]
            batch_pred_map_cls.append(cur_list)
        else:
            batch_pred_map_cls.append([(pred_sem_cls[i, j].item(), pred_corners_3d_upright_camera[i, j], obj_prob[i, j]) \
                                       for j in range(pred_center.shape[1]) if
                                       pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']])

    return batch_pred_map_cls, pred_mask


def parse_groundtruths(end_points, config_dict):
    """ Parse groundtruth labels to OBB parameters.
    
    Args:
        end_points: dict
            {center_label, heading_class_label, heading_residual_label,
            size_class_label, size_residual_label, sem_cls_label,
            box_label_mask}
        config_dict: dict
            {dataset_config}

    Returns:
        batch_gt_map_cls: a list  of len == batch_size (BS)
            [gt_list_i], i = 0, 1, ..., BS-1
            where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
            where j = 0, ..., num of objects - 1 at sample input i
    """
    center_label = end_points['center_label']
    heading_class_label = end_points['heading_class_label']
    heading_residual_label = end_points['heading_residual_label']
    size_class_label = end_points['size_class_label']
    size_residual_label = end_points['size_residual_label']
    box_label_mask = end_points['box_label_mask']
    sem_cls_label = end_points['sem_cls_label']
    bsize = center_label.shape[0]

    K2 = center_label.shape[1]  # K2==MAX_NUM_OBJ
    gt_corners_3d_upright_camera = np.zeros((bsize, K2, 8, 3))
    gt_center_upright_camera = flip_axis_to_camera(center_label[:, :, 0:3].detach().cpu().numpy())
    for i in range(bsize):
        for j in range(K2):
            if box_label_mask[i, j] == 0: continue
            heading_angle = config_dict['dataset_config'].class2angle(heading_class_label[i, j].detach().cpu().numpy(),
                                                                      heading_residual_label[
                                                                          i, j].detach().cpu().numpy())
            box_size = config_dict['dataset_config'].class2size(int(size_class_label[i, j].detach().cpu().numpy()),
                                                                size_residual_label[i, j].detach().cpu().numpy())
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, gt_center_upright_camera[i, j, :])
            gt_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

    batch_gt_map_cls = []
    for i in range(bsize):
        batch_gt_map_cls.append([(sem_cls_label[i, j].item(), gt_corners_3d_upright_camera[i, j]) for j in
                                 range(gt_corners_3d_upright_camera.shape[1]) if box_label_mask[i, j] == 1])
    end_points['batch_gt_map_cls'] = batch_gt_map_cls

    return batch_gt_map_cls


def get_verts(center,width,height,normal_vector):

    normal_vector = normal_vector/max(np.linalg.norm(normal_vector),1e-6)
    center = np.array(center)
    normal_vector = np.array(normal_vector)
    x1 = center[0] + width * normal_vector[1] /2
    x2 = center[0] - width * normal_vector[1] /2

    x=[x1,x2]

    y1 = center[1] - width * normal_vector[0] /2
    y2 = center[1] + width * normal_vector[0] /2

    y=[y1,y2]

    h1 = center[2] + height/2
    h2 = center[2] - height/2

    h=[h1,h2]

    corners = []

    for _ in h:
        corners.append([x1,y1,_])
        corners.append([x2,y2,_])

    return np.array(corners)

def get_verts_tensor(center, width, height, normal_vector):
    normal_vector = normal_vector / max(np.linalg.norm(normal_vector.cpu().detach()),1e-6)

    x1 = center[0] + width * normal_vector[1] /2
    x2 = center[0] - width * normal_vector[1] /2

    y1 = center[1] - width * normal_vector[0] /2
    y2 = center[1] + width * normal_vector[0] /2

    h1 = center[2] + height/2
    h2 = center[2] - height/2

    x, y, h = torch.stack([x1, x2]).cuda(), torch.stack([y1, y2]).cuda(), torch.stack([h1, h2]).cuda()

    corners = []

    corners.append(torch.stack([x1,y1,h1]))
    corners.append(torch.stack([x2,y2,h1]))
    corners.append(torch.stack([x1,y1,h2]))
    corners.append(torch.stack([x2,y2,h2]))

    return torch.stack(corners, dim=0)



def parse_quad_predictions(end_points, config_dict, prefix=""):
    """ Parse predictions to OBB parameters and suppress overlapping boxes
    
    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """

    pred_center = end_points[f'{prefix}quad_center']  # B,num_proposal,3
    pred_size = end_points[f'{prefix}quad_size']
    normal_vector =  end_points[f'{prefix}normal_vector']


    num_proposal = pred_center.shape[1]
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    # pred_size_check = end_points[f'{prefix}pred_size']  # B,num_proposal,3
    # pred_bbox_check = end_points[f'{prefix}bbox_check']  # B,num_proposal,3

    bsize = pred_center.shape[0]
    
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_corners_3d_upright_camera_tensor = torch.zeros(size=(bsize, num_proposal, 8, 3))

    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    pred_center_upright_camera_tensor = flip_axis_to_camera_tensor(pred_center)

    pred_corners = np.zeros((bsize, num_proposal, 4, 3))
    pred_corners_tensor = torch.zeros((bsize, num_proposal, 4, 3))
    
    for i in range(bsize):
        for j in range(num_proposal):

            # Numpy
            cos_theta = torch.cosine_similarity(torch.tensor(normal_vector[i,j,:].detach().cpu().numpy()),torch.tensor([0,1,0]),dim=0)
            heading_angle = torch.arccos(cos_theta)
            cos_theta1 = torch.cosine_similarity(torch.tensor(normal_vector[i,j,:].detach().cpu().numpy()),torch.tensor([1,0,0]),dim=0)
            if cos_theta1>0:
                heading_angle = np.pi*2 - heading_angle

            # Torch
            cos_theta_tensor = torch.cosine_similarity(normal_vector[i,j,:], torch.tensor([0,1,0]).cuda(), dim=0)
            heading_angle_tensor = torch.arccos(cos_theta_tensor)
            cos_theta1_tensor = torch.cosine_similarity(normal_vector[i,j,:], torch.tensor([1,0,0]).cuda(), dim=0)
            if cos_theta1_tensor > 0:
                heading_angle_tensor = torch.pi * 2 - heading_angle_tensor

            # Numpy
            width = pred_size[i,j,0].detach().cpu().numpy()
            height = pred_size[i,j,1].detach().cpu().numpy()
            box_size = np.array([width,LENGTH,height])
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, pred_center_upright_camera[i, j, :])
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

            pred_corners[i,j,:] = get_verts(pred_center[i,j,:].detach().cpu().numpy(),width,height,normal_vector[i,j,:].detach().cpu().numpy())

            # Torch
            width_tensor = pred_size[i, j, 0]
            height_tensor = pred_size[i, j, 1]
            box_size_tensor = torch.stack([width_tensor, torch.tensor(LENGTH).cuda(), height_tensor])
            corners_3d_upright_camera_tensor = get_3d_box_tensor(box_size_tensor, heading_angle_tensor, pred_center_upright_camera_tensor[i, j, :])
            pred_corners_3d_upright_camera_tensor[i, j] = corners_3d_upright_camera_tensor

            pred_corners_tensor[i, j, :] = get_verts_tensor(pred_center[i, j, :], width_tensor, height_tensor, normal_vector[i,j,:])


    K = pred_center.shape[1]  # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))

    obj_logits = end_points[f'{prefix}quad_scores'].detach().cpu().numpy()
    ## ???
    # obj_prob = sigmoid(obj_logits)[:, :, 1]  # (B,K)
    obj_prob = softmax(obj_logits)[:, :, 1]  # (B,K)

    obj_logits_tensor = end_points[f'{prefix}quad_scores']
    obj_prob_tensor = torch.sigmoid(obj_logits_tensor.reshape(-1, 2)).reshape(obj_logits_tensor.shape)
    #print(obj_prob)

    # ---------- NMS input: pred_with_prob in (B,K,7) -----------
    pred_mask = np.zeros((bsize, K))
    for i in range(bsize):
        boxes_3d_with_prob = np.zeros((K, 7))
        for j in range(K):
            boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
            boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
            boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
            boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
            boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
            boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
            boxes_3d_with_prob[j, 6] = obj_prob[i, j]
        nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
        
        try:
            quad_nms_thres = config_dict['nms_iou_quad']
        except:
            quad_nms_thres = config_dict['nms_iou']
        
        pick = nms_3d_faster(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                quad_nms_thres, config_dict['use_old_type_nms'])
        assert (len(pick) > 0)
        pred_mask[i, nonempty_box_inds[pick]] = 1
    # ---------- NMS output: pred_mask in (B,K) -----------

    batch_pred_map_cls = []  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    
    batch_pred_corners_list = []

    batch_pred_map_cls_tensor = [[] for _ in range(bsize)]
    batch_pred_corners_list_tensor = [[] for _ in range(bsize)]
    
    for i in range(bsize):
        batch_pred_map_cls.append([(1, pred_corners_3d_upright_camera[i, j], obj_prob[i, j]) \
                                    for j in range(pred_center.shape[1]) if pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']])
        batch_pred_corners_list.append([pred_corners[i, j]\
                                    for j in range(pred_center.shape[1]) if pred_mask[i, j] == 1 and obj_prob[i, j] > QUAD_THRES])

        for j in range(pred_center.shape[1]):
            if pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']:
                batch_pred_map_cls_tensor[i].append((1, pred_corners_3d_upright_camera_tensor[i, j], obj_prob_tensor[i, j]))
            if pred_mask[i, j] == 1 and obj_prob[i, j] > 0.5:
                batch_pred_corners_list_tensor[i].append(pred_corners_tensor[i, j])

    end_points[f"{prefix}batch_pred_map_cls_tensor"] = batch_pred_map_cls_tensor
    end_points[f"{prefix}batch_pred_corners_list_tensor"] = batch_pred_corners_list_tensor

    return batch_pred_map_cls,pred_mask, batch_pred_corners_list


def parse_quad_groundtruths(end_points, config_dict):
    """ Parse groundtruth labels to OBB parameters.
    
    Args:
        end_points: dict
            {center_label, heading_class_label, heading_residual_label,
            size_class_label, size_residual_label, sem_cls_label,
            box_label_mask}
        config_dict: dict
            {dataset_config}

    Returns:
        batch_gt_map_cls: a list  of len == batch_size (BS)
            [gt_list_i], i = 0, 1, ..., BS-1
            where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
            where j = 0, ..., num of objects - 1 at sample input i
    """
    center_label = end_points['gt_quad_centers']
    size_label = end_points['gt_quad_sizes']
    vector_label =  end_points['gt_normal_vectors']
    num_gt_quads = end_points['num_gt_quads']
    num_total_quads = end_points['num_total_quads']
    bsize = center_label.shape[0]

    K2 = MAX_NUM_QUAD #num_gt_quads  # K2==MAX_NUM_OBJ
    gt_corners_3d_upright_camera = np.zeros((bsize, K2, 8, 3))
    gt_center_upright_camera = flip_axis_to_camera(center_label[:, :, 0:3].detach().cpu().numpy())
    
    gt_corners = np.zeros((bsize, K2, 4, 3))
    
    for i in range(bsize):
        #K2 = num_gt_quads[i,0]
        for j in range(K2):
            cos_theta = torch.cosine_similarity(torch.tensor(vector_label[i,j,:].detach().cpu().numpy()),torch.tensor([0,1,0]),dim=0)
            heading_angle = torch.arccos(cos_theta)
            cos_theta1 = torch.cosine_similarity(torch.tensor(vector_label[i,j,:].detach().cpu().numpy()),torch.tensor([1,0,0]),dim=0)
            if cos_theta1>0:
              heading_angle = np.pi*2 - heading_angle 
            width = size_label[i,j,0].detach().cpu().numpy()
            height = size_label[i,j,1].detach().cpu().numpy()
            box_size = np.array([width,LENGTH,height])

            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, gt_center_upright_camera[i, j, :])
            gt_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

            gt_corners[i,j,:] = get_verts(center_label[i,j,:].detach().cpu().numpy(),width,height,vector_label[i,j,:].detach().cpu().numpy())

    batch_gt_map_cls = []
    batch_gt_corners_list = []

    for i in range(bsize):
        batch_gt_map_cls.append([(1, gt_corners_3d_upright_camera[i, j]) for j in
                                 range(gt_corners_3d_upright_camera.shape[1]) if ( j < num_gt_quads[i, j])])
        batch_gt_corners_list.append([gt_corners[i,j] for j in range(gt_corners.shape[1]) if ( j < num_total_quads[i, j])] )
    end_points['batch_gt_map_cls'] = batch_gt_map_cls

    return batch_gt_map_cls,batch_gt_corners_list


class APCalculator(object):
    ''' Calculating Average Precision '''

    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()

    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.
        
        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """

        bsize = len(batch_pred_map_cls)
        assert (bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.scan_cnt += 1

    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        rec, prec, ap = eval_det_multiprocessing(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh,
                                                 get_iou_func=get_iou_obb)
        ret_dict = {}
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision' % (clsname)] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall' % (clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall' % (clsname)] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)
        return ret_dict

    def reset(self):
        self.gt_map_cls = {}  # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {}  # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0


SAME_THRES = 0.40
class QUADAPCalculator(object):
    ''' Calculating Average Precision '''

    def __init__(self, ap_iou_thresh=0.25, class2type_map=None, logger=None, logger_i=None):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.logger = logger
        self.I = logger_i
        self.reset()

    def step(self, batch_pred_map_cls, batch_gt_map_cls,batch_pred_corners_list,batch_gt_corners_list,batch_gt_horizontal_list):
        """ Accumulate one batch of prediction and groundtruth.
        
        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """

        bsize = len(batch_pred_map_cls)
        assert (bsize == len(batch_gt_map_cls))

        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.pred_corners[self.scan_cnt] = batch_pred_corners_list[i]
            self.gt_corners[self.scan_cnt] = batch_gt_corners_list[i]
            self.horizontal_corners[self.scan_cnt] = batch_gt_horizontal_list[i]
            self.scan_cnt += 1


    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        rec, prec, ap = eval_det_multiprocessing(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh,
                                                 get_iou_func=get_iou_obb)
        ret_dict = {}
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision' % (clsname)] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall' % (clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall' % (clsname)] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)
        return ret_dict

    def reset(self):
        self.gt_map_cls = {}  # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {}  # {scan_id: [(classname, bbox, score)]}
        self.pred_corners = {}  
        self.gt_corners = {}
        self.horizontal_corners = {}
        self.scan_cnt = 0


    def same_point(self,pred,gt):
        distance = np.linalg.norm(np.array(pred-gt))
        return (distance <= SAME_THRES)
        

    def compute_correctness(self,pred_corner,all_gt,is_embed=False):
        for gt in all_gt:
            correctness1, correctness2 = True, True
            for i in range(0,4):
                distance = np.linalg.norm(np.array(pred_corner[i]-gt[i]))
                if distance > SAME_THRES:
                    correctness1 = False
            for i in range(0,4):
                distance = np.linalg.norm(np.array(pred_corner[i]-gt[i^1]))  # [0, 1, 2, 3] - [1, 0, 3, 2], ignore positive or negative normals
                if distance > SAME_THRES:
                    correctness2 = False
            if correctness1 or correctness2:
                return True
        return False

    def contain_point(self,pointlist,point):
        for p in pointlist:
            if self.same_point(p,point):
                return True,p
        return False,None

    def get_ceiling_and_floor(self, pred_corners):
        ceilings = []
        floors = []
        for quad_corner in pred_corners:
            for i in range(0,2):
                contain,p = self.contain_point(ceilings,quad_corner[i])
                if not contain:
                    ceilings.append(quad_corner[i])
                else:
                    new_corner = (p + quad_corner[i])/2
                    ceilings.append(new_corner)

            for i in range(2,4):
                contain,p = self.contain_point(floors,quad_corner[i])
                if not contain:
                    floors.append(quad_corner[i])
                else:
                    new_corner = (p + quad_corner[i])/2
                    floors.append(new_corner)

        return ceilings, floors

    def compute_F1(self, calculated = False, is_ema=False):
        """
        find point radius < SAME_THRES
        """
        tp = 0
        fn = 0
        fp = 0
        
        npos=0
        for i in range(0,self.scan_cnt):
            npos += len(self.gt_corners[i])


        for i in range(0,self.scan_cnt):

            all_pred_corners = self.pred_corners[i] 
            all_gt_corners = self.gt_corners[i]
            horizontal_quads  = self.horizontal_corners[i]

            horizontal_quads = np.array(horizontal_quads.cpu())

            for pred_corner in all_pred_corners:
                if self.compute_correctness(pred_corner,all_gt_corners):
                    tp = tp + 1
                else:
                    fp = fp + 1
            
            if calculated == True: #calculate horizontal quads
                ceilings, floors = self.get_ceiling_and_floor(all_pred_corners)
                if len(ceilings)==4:
                    if self.compute_correctness(ceilings, horizontal_quads,True):
                        tp = tp + 1
                if len(floors)==4:
                    if self.compute_correctness(floors, horizontal_quads):
                        tp = tp + 1

        p = tp/max((tp+fp),1e-6)
        #r = tp/(tp+fn)
        r = tp/npos
        
        f1 = 2.0*p*r/max((p+r),1e-6)
        
        if self.logger is not None:
            self.logger.info(f"{tp}, {fp}, {fn}, {npos}")
            if is_ema:
                import wandb
                wandb.log({
                    "val-ema/tp": tp,
                    "val-ema/fp": fp,
                    "val-ema/F1-p": p,
                    "val-ema/F1-r": r,
                    "val-ema/F1": f1
                }, step=self.I)
            else:
                import wandb
                wandb.log({
                    "val/tp": tp,
                    "val/fp": fp,
                    "val/F1-p": p,
                    "val/F1-r": r,
                    "val/F1": f1
                }, step=self.I)

        return f1

        

            


