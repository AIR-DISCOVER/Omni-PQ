# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import os
import random
import sys

import numpy as np
from torch.utils.data import Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import pc_util
from scannet.model_util_scannet import rotate_aligned_boxes,rotate_quad

from scannet.model_util_scannet import ScannetDatasetConfig
from scannet.scannet_planes import get_quads

DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 64
MAX_NUM_QUAD = 32
NUM_PROPOSAL = 256
NUM_QUAD_PROPOSAL = 256

MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

class ScannetDetectionDataset(Dataset):
       
    def __init__(self, split_set='train', num_points=40000,
        use_color=False, use_height=False, augment=False,
                 start_proportion=0.0, end_proportion=1.0,):

        self.data_path = os.path.join(BASE_DIR, 'scannet_train_detection_data')
        self.data_path_with_planes = os.path.join(BASE_DIR, 'scannet_planes')

        all_scan_names = list(set([os.path.basename(x)[0:12] \
            for x in os.listdir(self.data_path_with_planes) if x.startswith('scene')]))
        if split_set=='all':            
            self.scan_names = all_scan_names
        elif split_set in ['train', 'val']:
            split_filenames = os.path.join(ROOT_DIR, 'scannet/meta_data',
                'scannetv2_{}.txt'.format(split_set))
            
            self.get_quads_fn = get_quads

            with open(split_filenames, 'r') as f:
                self.scan_names = f.read().splitlines()

            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [sname for sname in self.scan_names \
                if sname in all_scan_names]
            
            # Here change the dataset to splitted
            self.scan_names = sorted(self.scan_names)
            bak_scan_names = self.scan_names
            
            self.start_idx = int(len(self.scan_names) * start_proportion)
            self.end_idx = int(len(self.scan_names) * end_proportion)
            self.scan_names = self.scan_names[self.start_idx:self.end_idx]
            if len(self.scan_names) == 0:
                self.scan_names = [bak_scan_names[-1], ]
            
            print('kept {} scans out of {}'.format(len(self.scan_names), num_scans))
        else:
            print('illegal split name')
            return
        
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.augment = augment
       
    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx, **kwargs):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            angle_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            angle_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            point_votes: (N,3) with votes XYZ
            point_votes_mask: (N,) with 0/1 with 1 indicating the point is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            pcl_color: unused
        """
        # print(idx, len(self.scan_names))
        scan_name = self.scan_names[idx]
        if "scan_name" in kwargs:
            scan_name = kwargs['scan_name']
        mesh_vertices = np.load(os.path.join(self.data_path, scan_name)+'_vert.npy')
        instance_labels = np.load(os.path.join(self.data_path, scan_name)+'_ins_label.npy')
        semantic_labels = np.load(os.path.join(self.data_path, scan_name)+'_sem_label.npy')
        instance_bboxes = np.load(os.path.join(self.data_path, scan_name)+'_bbox.npy')
        vertex_normals = np.load(os.path.join(self.data_path, os.pardir, "scannet_train_detection_data_normals", scan_name) + ".normal.npy")
        # TODO: [c7w] Remove this
        # point_guide_labels = np.load(os.path.join(self.data_path, os.pardir, "scannet_train_layout_data_points", scan_name) + '.npy')
        # point_guide_label_normals = np.load(os.path.join(self.data_path, os.pardir, "scannet_train_layout_data_points", scan_name) + '.normal.npy')

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_COLOR_RGB)/256.0
        
        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
            
        # ------------------------------- LABELS ------------------------------        
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        size_gts = np.zeros((MAX_NUM_OBJ, 3))

        ema_point_clouds, ema_choices = pc_util.random_sampling(point_cloud,
            self.num_points, return_choices=True)
        point_cloud, choices = pc_util.random_sampling(point_cloud,
            self.num_points, return_choices=True)        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        vertex_normals = vertex_normals[choices]
        
        pcl_color = pcl_color[choices]
        
        target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
        target_bboxes[0:instance_bboxes.shape[0],:] = instance_bboxes[:,0:6]

        rectangles,total_quad_num, horizontal_quads = self.get_quads_fn(scan_name)

        
        # ------------------------------- DATA AUGMENTATION ------------------------------    
        flip_YZ_XZ = np.array([False, False])
        rot_mat = np.identity(3)
        scale_ratio = np.array(1.0)
        if self.augment:
            
            # Switches for ablation study: data augmentation
            DISABLE_DOWNSAMPLE = 0
            DISABLE_FLIP = 0
            DISABLE_ROT = 0
            DISABLE_SCALING = 0
            
            if DISABLE_DOWNSAMPLE:
                ema_point_clouds = point_cloud
            
            if not DISABLE_FLIP:
                if np.random.random() > 0.5:
                    # Flipping along the YZ plane
                    flip_YZ_XZ[0] = True
                    point_cloud[:,0] = -1 * point_cloud[:,0]
                    target_bboxes[:,0] = -1 * target_bboxes[:,0]
                    vertex_normals[:, 0] = -1 * vertex_normals[:, 0]
                    # point_guide_labels[..., 0] = -1 * point_guide_labels[..., 0]
                    # point_guide_label_normals[..., 0] = -1 * point_guide_label_normals[..., 0]
                    rectangles[:,0] = -1* rectangles[:,0]      
                    rectangles[:,3] = -1* rectangles[:,3]
                    if horizontal_quads.shape[0] > 0:
                        horizontal_quads[..., 0] = -1 * horizontal_quads[..., 0]
                    
                if np.random.random() > 0.5:
                    # Flipping along the XZ plane
                    flip_YZ_XZ[1] = True
                    point_cloud[:,1] = -1 * point_cloud[:,1]
                    target_bboxes[:,1] = -1 * target_bboxes[:,1]
                    vertex_normals[:, 1] = -1 * vertex_normals[:, 1]
                    # point_guide_labels[..., 1] = -1 * point_guide_labels[..., 1]
                    # point_guide_label_normals[..., 1] = -1 * point_guide_label_normals[..., 1]
                    rectangles[:,1] = -1* rectangles[:,1]      
                    rectangles[:,4] = -1* rectangles[:,4]       
                    if horizontal_quads.shape[0] > 0:
                        horizontal_quads[..., 1] = -1 * horizontal_quads[..., 1]                     
            
            # Rotation along up-axis/Z-axis
            rot_angle = ((np.random.random()*np.pi/18) - np.pi/36) * 1 # -5 ~ +5 degree
            rot_angle += random.choice([0, 1, 2, 3]) * np.pi / 2
            if 'rot_angle' in kwargs:
                rot_angle = kwargs['rot_angle']
            if DISABLE_ROT:
                rot_angle = np.array(0.)
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            vertex_normals[:, 0:3] = np.dot(vertex_normals[:, 0:3], np.transpose(rot_mat))
            # point_guide_labels[..., 0:3] = np.dot(point_guide_labels[..., 0:3], np.transpose(rot_mat))
            # point_guide_label_normals[..., 0:3] = np.dot(point_guide_label_normals[:, 0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)
            rectangles = rotate_quad(rectangles, rot_mat)
            if horizontal_quads.shape[0] > 0:
                horizontal_quads[..., 0:3] = np.dot(horizontal_quads[..., 0:3], np.transpose(rot_mat))
            
            # Augment point cloud scale: 0.85x-1.15x
            # Assertion: Augment scales does not change normals
            scale_ratio = np.random.random() * 0.3 + 0.85
            if 'scale_ratio' in kwargs:
                scale_ratio = kwargs['scale_ratio']
            if DISABLE_SCALING:
                scale_ratio = np.array(1.)
            # scale_ratio = np.expand_dims(np.tile(scale_ratio_, 3), 0)
            point_cloud[:, 0:3] *= scale_ratio
            # point_guide_labels[..., 0:3] *= scale_ratio
            target_bboxes[:, 0:3] *= scale_ratio
            target_bboxes[:, 3:6] *= scale_ratio
            rectangles[:, 0:3] *= scale_ratio
            rectangles[:, 6:8] *= scale_ratio
            if horizontal_quads.shape[0] > 0:
                horizontal_quads[..., 0:3] *= scale_ratio
            if self.use_height:
                point_cloud[:, -1] *= scale_ratio

        gt_centers = target_bboxes[:, 0:3]
        gt_centers[instance_bboxes.shape[0]:, :] += 1000.0
        
        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered 
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label. 
        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)
        point_instance_label = np.zeros(self.num_points) - 1
        
        for i_instance in np.unique(instance_labels):            
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label            
            if semantic_labels[ind[0]] in DC.nyu40ids:
                x = point_cloud[ind,:3]
                center = 0.5*(x.min(0) + x.max(0))

                ilabel = np.argmin(((center - gt_centers) ** 2).sum(-1))
                point_instance_label[ind] = ilabel
                
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3))  # make 3 votes identical 
        
        class_ind = [np.where(DC.nyu40ids == x)[0][0] for x in instance_bboxes[:,-1]]   
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:instance_bboxes.shape[0]] = class_ind
        size_residuals[0:instance_bboxes.shape[0], :] = \
            target_bboxes[0:instance_bboxes.shape[0], 3:6] - DC.mean_size_arr[class_ind,:]
        size_gts[0:instance_bboxes.shape[0], :] = target_bboxes[0:instance_bboxes.shape[0], 3:6]    
        
        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['vertex_normals'] = vertex_normals.astype(np.float32)
        ret_dict['semantic_labels'] = semantic_labels.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:,0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        ret_dict['size_gts'] = size_gts.astype(np.float32)
        num_gt_boxes = np.zeros((NUM_PROPOSAL))+ instance_bboxes.shape[0]
        ret_dict['num_gt_boxes'] =  num_gt_boxes.astype(np.int64)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))                                
        target_bboxes_semcls[0:instance_bboxes.shape[0]] = \
            [DC.nyu40id2class[x] for x in instance_bboxes[:,-1][0:instance_bboxes.shape[0]]]                
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['pcl_color'] = pcl_color
        ret_dict['point_instance_label'] = point_instance_label.astype(np.int64)
        ret_dict['use_gt'] = np.array(self.start_idx == 0).astype(np.bool)

        # Augmentation properties
        ret_dict['ema_point_clouds'] = ema_point_clouds.astype(np.float32)
        ret_dict['flip_x_axis'] = np.array(flip_YZ_XZ)[..., 0].astype(np.int64)
        ret_dict['flip_y_axis'] =  np.array(flip_YZ_XZ)[..., 1].astype(np.int64)
        ret_dict['rot_mat'] =  rot_mat.astype(np.float32)
        ret_dict['scale'] = np.array(scale_ratio).astype(np.float32)
    
        target_quad_centers = np.zeros((MAX_NUM_QUAD,3)) 
        target_normal_vectors = np.zeros((MAX_NUM_QUAD,3)) 
        target_quad_sizes = np.zeros((MAX_NUM_QUAD,2))    
        #target_quad_direction = np.zeros((MAX_NUM_QUAD,1))                      

        if rectangles.shape[0]>0:                                                
            target_quad_centers[0:rectangles.shape[0],:]=rectangles[:,0:3]         
            target_normal_vectors[0:rectangles.shape[0],:]=rectangles[:,3:6]
            target_quad_sizes[0:rectangles.shape[0],:]=rectangles[:,6:8]
            #target_quad_direction[0:rectangles.shape[0],:]=rectangles[:,8:9]

        ret_dict['gt_quad_centers'] = target_quad_centers.astype(np.float32)
        ret_dict['gt_quad_sizes'] = target_quad_sizes.astype(np.float32)
        # ret_dict['gt_quad_label'] = point_guide_labels.astype(np.float32)
        # ret_dict['gt_quad_label_normals'] = point_guide_label_normals.astype(np.float32)
        ret_dict['gt_normal_vectors'] = target_normal_vectors.astype(np.float32)
        #ret_dict['gt_quad_direction'] = target_quad_direction.astype(np.float32)

        num_gt_quads = np.zeros((NUM_QUAD_PROPOSAL))+ rectangles.shape[0]
        ret_dict['num_gt_quads'] =  num_gt_quads.astype(np.int64)
        num_total_quads = np.zeros((NUM_QUAD_PROPOSAL)) + total_quad_num
        ret_dict['num_total_quads'] =  num_total_quads.astype(np.int64)

        ret_dict['scan_name'] = scan_name
        target_horizontal_quads= np.zeros((4,4,3))
        if len(horizontal_quads)>0:
            target_horizontal_quads[0:len(horizontal_quads),:]=horizontal_quads[:,:,:]                                              
        ret_dict['horizontal_quads'] = target_horizontal_quads.astype(np.float32)

        return ret_dict
        
############# Visualizaion ########

def viz_votes(pc, point_votes, point_votes_mask, name=''):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask==1)
    pc_obj = pc[inds,0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds,0:3]    
    pc_util.write_ply(pc_obj, 'pc_obj{}.ply'.format(name))
    pc_util.write_ply(pc_obj_voted1, 'pc_obj_voted1{}.ply'.format(name))
    
def viz_obb(pc, label, mask, angle_classes, angle_residuals,
    size_classes, size_residuals, name=''):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ
    mask: (K,)
    angle_classes: (K,)
    angle_residuals: (K,)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0: continue
        obb = np.zeros(7)
        obb[0:3] = label[i,0:3]
        heading_angle = 0 # hard code to 0
        box_size = DC.mean_size_arr[size_classes[i], :] + size_residuals[i, :]
        obb[3:6] = box_size
        obb[6] = -1 * heading_angle
        print(obb)        
        oriented_boxes.append(obb)
    pc_util.write_oriented_bbox(oriented_boxes, 'gt_obbs{}.ply'.format(name))
    pc_util.write_ply(label[mask==1,:], 'gt_centroids{}.ply'.format(name))


#### Debugging ####

if __name__=='__main__': 
    dset = ScannetDetectionDataset(use_height=True, num_points=40000)
    
    item = dset.__getitem__(0, scan_name="scene0046_02")
    pc_util.write_oriented_bbox_with_normal(item['gt_quad_centers'])

    # for i_example in range(4):
    #     example = dset.__getitem__(i_example)
    #     print(example['gt_normal_vectors'])
    #     """
    #     pc_util.write_ply(example['point_clouds'], 'pc_{}.ply'.format(i_example))    
    #     viz_votes(example['point_clouds'], example['vote_label'],
    #         example['vote_label_mask'],name=i_example)    
    #     viz_obb(pc=example['point_clouds'], label=example['center_label'],
    #         mask=example['box_label_mask'],
    #         angle_classes=None, angle_residuals=None,
    #         size_classes=example['size_class_label'], size_residuals=example['size_residual_label'],
    #         name=i_example)
    #     """