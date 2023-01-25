import numpy as np
import torch
import os
import sys
from models.ap_helper_pq import QUADAPCalculator, parse_quad_groundtruths, parse_quad_predictions
from scannet.model_util_scannet import ScannetDatasetConfig
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util

DUMP_CONF_THRESH = 0.995  # Dump boxes with obj prob larger than that.
LENGTH = 0.15
SINGLE = True

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def dump_single_quad(quad_center, normal_vector, quad_size, quad_score, dump_dir, config):
    import os, pc_util
    # Generate OBB    
    cos_theta = torch.cosine_similarity(torch.tensor(normal_vector).cpu(), torch.tensor([0,1,0]),dim=0)
    heading_angle = torch.arccos(cos_theta) 
    cos_theta1 = torch.cosine_similarity(torch.tensor(normal_vector).cpu(), torch.tensor([1,0,0]),dim=0)
    if cos_theta1 > 0:
        heading_angle = np.pi*2 - heading_angle               
    obb = np.zeros((7,))
    obb[0:3] = quad_center.detach().cpu()
    obb[3] = quad_size[0].detach().cpu()
    obb[4] = LENGTH
    obb[5] = quad_size[1].detach().cpu()
    obb[6] = heading_angle

    def get_color(p):
        if p < 0.5:
            hue = p * 2
            return (hue, 0., hue, 1.0)
        else:
            hue = (p - 0.5) * 2
            return (1-hue, 0., 1, 1.0)

    colors = [get_color(quad_score), ]

    pc_util.write_oriented_bbox([obb, ], 
                                os.path.join(dump_dir, 'mixture_quad_result.ply'),
                                colors=colors)


def dump_results_quad(end_points, dump_dir, config, inference_switch=False, CONFIG_DICT=None, raw_colors=None):
    ''' Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    '''
    
    batch_pred_quad_map_cls,pred_quad_mask,batch_pred_quad_corner = parse_quad_predictions(end_points, CONFIG_DICT, "last_")
    
    dump_gt = ("gt_quad_centers" in end_points)
    if dump_gt:
        batch_gt_quad_map_cls,batch_gt_quad_corner = parse_quad_groundtruths(end_points, CONFIG_DICT)
    
    end_points['pred_quad_mask'] = pred_quad_mask
    
    if not os.path.exists(dump_dir):
        os.system('mkdir %s'%(dump_dir))

    # INPUT
    point_clouds = end_points['point_clouds'].cpu().numpy()
    batch_size = point_clouds.shape[0]


    objectness_scores = end_points['last_quad_scores'].detach().cpu().numpy() # (B,K,2)
    pred_center = end_points['last_quad_center'].detach().cpu().numpy() # (B,K,3)
    normal_vector =  end_points['last_normal_vector'].detach().cpu().numpy()
    pred_size = end_points['last_quad_size'].detach().cpu().numpy()

    scan_names = end_points['scan_name']
    aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
 
    # OTHERS
    if 'pred_quad_mask' in end_points:
        pred_mask = end_points['pred_quad_mask']
    elif 'pred_mask' in end_points:
        pred_mask = end_points['pred_mask'] # B,num_proposal
    else:
        pred_mask = None
    idx_beg = 0

    for i in range(batch_size):
        pc = point_clouds[i,:,:]
        objectness_prob = softmax(objectness_scores[i,:,:])[:,1] # (K,)

        # Dump various point clouds
        #pc_util.write_ply(pc, os.path.join(dump_dir, '%s_pc.ply'%(scan_names[i])))
        #pc_util.write_ply(aggregated_vote_xyz[i,:,:], os.path.join(dump_dir, '%s_aggregated_fps_pc.ply'%(scan_names[i])))
        # Dump predicted bounding boxes
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>=0:
            num_proposal = pred_center.shape[1]
            obbs = []
            for j in range(num_proposal):
                cos_theta = torch.cosine_similarity(torch.tensor(normal_vector[i,j,:]),torch.tensor([0,1,0]),dim=0)
                heading_angle = torch.arccos(cos_theta) 
                cos_theta1 = torch.cosine_similarity(torch.tensor(normal_vector[i,j,:]),torch.tensor([1,0,0]),dim=0)
                if cos_theta1>0:
                  heading_angle = np.pi*2 - heading_angle               
                obb = np.zeros((7,))
                obb[0:3] = pred_center[i,j]
                obb[3] = pred_size[i,j,0]
                obb[4] = LENGTH
                obb[5] = pred_size[i,j,1]
                obb[6] = heading_angle
                obbs.append(obb)
            
            if len(obbs)>=0:
                obbs = np.vstack(tuple(obbs)) # (num_proposal, 7)
                
                TP, FP = 0, 0
                
                if pred_mask is not None:
                    # Write pred_confident_nms_quad, green for TP, yellow for FP
                    boxes, colors = [], []
                    filtered_boxes = obbs[np.logical_and(objectness_prob>DUMP_CONF_THRESH, pred_mask[i,:]==1),:]
                    if len(filtered_boxes) != len(batch_pred_quad_corner[i]):
                        import IPython
                        IPython.embed()
                    for kk, box in enumerate(filtered_boxes):
                        boxes.append(box)
                        if dump_gt:
                            class2quad = ScannetDatasetConfig().class2quad
                            is_TP = QUADAPCalculator(ap_iou_thresh=0.25, class2type_map=class2quad).compute_correctness(
                                batch_pred_quad_corner[i][kk], batch_gt_quad_corner[i]
                            )
                            colors.append((0., 1., 0., 0.) if is_TP else (1., 1., 0., 0.))
                            if is_TP:
                                TP += 1
                            else:
                                FP += 1
                        else:
                            colors.append((0.6, 0.6, 0.6))
                    if SINGLE:
                        os.makedirs(os.path.join(dump_dir, "%s_confident_nms_quad_TP%dFP%d" % (scan_names[i], TP, FP)), exist_ok=True)
                        pc_util.write_oriented_bbox(boxes, 
                                                os.path.join(dump_dir, "%s_confident_nms_quad_TP%dFP%d" % (scan_names[i], TP, FP), \
                                                    '%s_pred_confident_nms_quad_TP%dFP%d.ply' % (scan_names[i], TP, FP)), 
                                                colors=colors, single=SINGLE)
                    else:
                        pc_util.write_oriented_bbox(boxes, 
                                                os.path.join(dump_dir, '%s_pred_confident_nms_quad_TP%dFP%d.ply' % (scan_names[i], TP, FP)), 
                                                colors=colors, single=SINGLE)
                    
                    # Write pred_confident_quad
                    def get_color(p):
                        # 0.5 <= p <= 1.0
                        hue = (p - 0.5) * 2
                        return (0., 0., hue, 1.0)
                    boxes, colors = [], []
                    filtered_obbs = obbs[objectness_prob>DUMP_CONF_THRESH, :]
                    filtered_possibilities = objectness_prob[objectness_prob>DUMP_CONF_THRESH]
                    for kk, box in enumerate(filtered_obbs):
                        p = filtered_possibilities[kk]
                        boxes.append(box)
                        colors.append(get_color(p))
                    if SINGLE:
                        os.makedirs(os.path.join(dump_dir, f"{scan_names[i]}_confident"), exist_ok=True)
                        pc_util.write_oriented_bbox(boxes, 
                                                os.path.join(dump_dir, f"{scan_names[i]}_confident", '%s_pred_confident_quad.ply'%(scan_names[i])),
                                                colors=colors, single=SINGLE)
                    else:
                        pc_util.write_oriented_bbox(boxes, 
                                                os.path.join(dump_dir, '%s_pred_confident_quad.ply'%(scan_names[i])),
                                                colors=colors, single=SINGLE)
                    
                    # Write pred_nms_quad: 0~0.5 purple, 0.5~1.0 blue
                    def get_color2(p):
                        if p < 0.5:
                            hue = p * 2
                            return (hue, 0., hue, 1.0)
                        else:
                            hue = (p - 0.5) * 2
                            return (1-hue, 0., 1, 1.0)
                    boxes, colors = [], []
                    filtered_obbs = obbs[pred_mask[i,:]==1,:]
                    filtered_possibilities = objectness_prob[pred_mask[i,:]==1]
                    for kk, box in enumerate(filtered_obbs):
                        p = filtered_possibilities[kk]
                        boxes.append(box)
                        colors.append(get_color2(p))
                    if SINGLE:
                        os.makedirs(os.path.join(dump_dir, f"{scan_names[i]}_nms"), exist_ok=True)
                        pc_util.write_oriented_bbox(boxes, 
                                                os.path.join(dump_dir, f"{scan_names[i]}_nms", '%s_pred_nms_quad.ply'%(scan_names[i])),
                                                colors=colors, single=SINGLE)
                    else:
                        pc_util.write_oriented_bbox(boxes, 
                                                os.path.join(dump_dir, '%s_pred_nms_quad.ply'%(scan_names[i])),
                                                colors=colors, single=SINGLE)
                    
                # Write all quads
                if raw_colors is not None:
                    boxes, colors = obbs, raw_colors
                    # pc_util.write_oriented_bbox(boxes, 
                    #                         os.path.join(dump_dir, '%s_all_quad.ply'%(scan_names[i])),
                    #                         colors=colors)


    # Return if it is at inference time. No dumping of groundtruths
    if inference_switch:
        return

    if dump_gt:

        # LABELS
        center_label = end_points['gt_quad_centers'].detach().cpu().numpy() 
        size_label = end_points[f'gt_quad_sizes'].detach().cpu().numpy() 
        vector_label =  end_points[f'gt_normal_vectors'].detach().cpu().numpy() 
        num_gt_quads = end_points['num_gt_quads'].detach().cpu().numpy() 

        for i in range(batch_size):
            # Dump GT bounding boxes
            obbs = []
            for j in range(num_gt_quads[i,0]):
                cos_theta = torch.cosine_similarity(torch.tensor(vector_label[i,j,:]),torch.tensor([0,1,0]),dim=0)
                heading_angle = torch.arccos(cos_theta)  
                cos_theta1 = torch.cosine_similarity(torch.tensor(vector_label[i,j,:]),torch.tensor([1,0,0]),dim=0)
                if cos_theta1>0:
                    heading_angle = np.pi*2 - heading_angle               
                obb = np.zeros((7,))
                obb[0:3] = center_label[i,j]
                obb[3] = size_label[i,j,0]
                obb[4] = LENGTH
                obb[5] = size_label[i,j,1]
                obb[6] = heading_angle
                obbs.append(obb)
            if len(obbs)>0:
                obbs = np.vstack(tuple(obbs)) # (num_gt_objects, 7)
                pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%s_gt_quad_GT%d.ply'%(scan_names[i], obbs.shape[0])),
                                            colors=list([(1., 0., 0., 1.) for k in range(obbs.shape[0])]))
        