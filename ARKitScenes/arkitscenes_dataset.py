import os
import random
import sys
import json

import IPython
import numpy as np
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from scannet.model_util_scannet import ScannetDatasetConfig
import utils.pc_util as pc_util
from models.dump_helper import dump_pc
from scannet.model_util_scannet import rotate_aligned_boxes,rotate_quad
# from model_util_scannet import ScannetDatasetConfig

import ARKitScenes.arkitscenes_utils as arkitscenes_utils

DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 64
MAX_NUM_QUAD = 32
NUM_PROPOSAL = 256
NUM_QUAD_PROPOSAL = 256

type2class = {
    "cabinet": 0, "refrigerator": 12, "shelf": 7, "stove": -1, "bed": 1, # 0..5
    "sink": 15, "washer": -1, "toilet": 14, "bathtub": 16, "oven": -1, # 5..10
    "dishwasher": -1, "fireplace": -1, "stool": -1, "chair": 2, "table": 4, # 10..15
    "tv_monitor": -1, "sofa": 3, # 15..17
}

def is_valid_mapping_name(mapping_name):
    mapping_file = os.path.join(BASE_DIR, "data", "annotations", f"{mapping_name}.json")
    if os.stat(mapping_file).st_size < 60:
        return False
    return True


class ARKitSceneDataset(Dataset):
    def __init__(self, split_set='train', num_points=40000,
        augment=False, start_proportion=0.0, end_proportion=1.0,):
        self.data_path = os.path.join(BASE_DIR, 'dataset')
        
        assert split_set in ['train', 'valid']
        self.split_set = split_set
        
        with open(os.path.join(self.data_path, f"{split_set}_filtered.txt"), 'r') as f:
            split_filenames = f.read().strip().split('\n')
       
        if split_set == "train":
            self.data_path = os.path.join(self.data_path, "3dod/Training")
        else:
            self.data_path = os.path.join(self.data_path, "3dod/Validation")
            self.valid_mapping = {line.split(",")[0]: line.split(",")[1] \
                                  for line in open(os.path.join(BASE_DIR, 'data', "file.txt")).read().strip().split("\n")}
        
        self.scan_names = sorted(split_filenames)
        bak_scan_names = self.scan_names
        
        self.start_idx = int(len(self.scan_names) * start_proportion)
        self.end_idx = int(len(self.scan_names) * end_proportion)
        self.scan_names = self.scan_names[self.start_idx:self.end_idx]

        # TODO: filter out unlabelled layout in valid set
        if self.split_set == "valid":
            self.scan_names = [scan_name for scan_name in self.scan_names if is_valid_mapping_name(self.valid_mapping[scan_name])]
    
        if len(self.scan_names) == 0:
            self.scan_names = [bak_scan_names[-1], ]
        
        print(f"Find {len(self.scan_names)} in ARKitScene dataset!")
        
        self.num_points = num_points
        self.augment = augment
    
    
    def __len__(self):
        return len(self.scan_names)
    
    def __getitem__(self, idx, **kwargs):
        scan_name = self.scan_names[idx]
        scan_dir = os.path.join(self.data_path, scan_name, f"{scan_name}_offline_prepared_data")
        mesh_vertices = np.load(os.path.join(scan_dir, f"{scan_name}_pc.npy"))
        vertex_normals = np.load(os.path.join(scan_dir, f"{scan_name}_normal.npy"))
        instance_bboxes = np.load(os.path.join(scan_dir, f"{scan_name}_bbox.npy"), allow_pickle=True).item()
    
        # Prepare label containers
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        size_gts = np.zeros((MAX_NUM_OBJ, 3))
            
        
        # TODO: OBB-Guided Scene Axis-Alignment
        angle = np.percentile(instance_bboxes['bboxes'][..., -1] % (np.pi / 2), 50)
        rot_mat = pc_util.rotz(angle)
        mesh_vertices_prime = mesh_vertices
        vertex_normals = vertex_normals
        
        z_filter_L = np.percentile(mesh_vertices_prime[..., 2], 15)
        z_filter_H = np.percentile(mesh_vertices_prime[..., 2], 85)
        filter_mask = (mesh_vertices_prime[..., 2] >= z_filter_L) & (mesh_vertices_prime[..., 2] <= z_filter_H)
        x_base = np.percentile(mesh_vertices_prime[filter_mask, 0], 50)
        y_base = np.percentile(mesh_vertices_prime[filter_mask, 1], 50)
        z_base = np.percentile(mesh_vertices_prime[..., 2], 5)
        offset = np.array([x_base, y_base, z_base])
        
        # mesh_vertices_prime = mesh_vertices_prime - offset
        
        instance_bboxes['bboxes'][..., :3] = np.dot(instance_bboxes['bboxes'][..., :3], np.transpose(rot_mat))
        instance_bboxes['bboxes'][..., :3] = instance_bboxes['bboxes'][..., :3] - offset
        instance_bboxes['bboxes'][..., 6] -= angle
        instance_bboxes['bboxes'][..., 6] %= 2 * np.pi
        reverse_mask = ((np.pi / 4 <= instance_bboxes['bboxes'][..., 6]) & (instance_bboxes['bboxes'][..., 6] <= np.pi / 4 * 3)) | \
            ((np.pi / 4 * 5 <= instance_bboxes['bboxes'][..., 6]) & (instance_bboxes['bboxes'][..., 6] <= np.pi / 4 * 7))
        dx = np.copy(instance_bboxes['bboxes'][..., 3])
        dy = np.copy(instance_bboxes['bboxes'][..., 4])
        instance_bboxes['bboxes'][..., 3] = dy * reverse_mask + dx * (1-reverse_mask)
        instance_bboxes['bboxes'][..., 4] = dx * reverse_mask + dy * (1-reverse_mask)
        # dump_pc(mesh_vertices_prime, dump_name="../dump/pc.txt", normal=None)
        # pc_util.write_bbox(instance_bboxes['bboxes'][..., :6], "../dump/box.ply")
        
        bbox_num = min(instance_bboxes['bboxes'].shape[0], MAX_NUM_OBJ)
        target_bboxes[0:bbox_num, :] = instance_bboxes['bboxes'][:, 0:6]
        target_bboxes_mask[0:bbox_num] = 1
        for n_bbox in range(bbox_num):
            str_type = instance_bboxes['types'][n_bbox]
            sem_id = type2class[str_type]
            target_bboxes_semcls[n_bbox] = sem_id
        num_gt_boxes = np.zeros((NUM_PROPOSAL)) + bbox_num

        # downsample points
        point_cloud, choices = pc_util.random_sampling(mesh_vertices_prime,
            self.num_points, return_choices=True)    
        selected_vertex_normals = vertex_normals[choices, ...]
        
        ema_point_clouds, ema_choices = pc_util.random_sampling(mesh_vertices_prime,
            self.num_points, return_choices=True)
        ema_vertex_normals = vertex_normals[ema_choices, ...]
        
        # data augmentation
        flip_YZ_XZ = np.array([False, False])
        rot_mat = np.identity(3)
        scale_ratio = np.array(1.0)
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                flip_YZ_XZ[0] = True
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                target_bboxes[:, 0] = -1 * target_bboxes[:, 0]
                vertex_normals[:, 0] = -1 * vertex_normals[:, 0]
            
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                flip_YZ_XZ[0] = False
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                target_bboxes[:, 1] = -1 * target_bboxes[:, 1]
                vertex_normals[:, 1] = -1 * vertex_normals[:, 1]
            
            # Rotation along up-axis / Z-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36
            rot_angle += random.choice([0, 1, 2, 3]) * np.pi / 2
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            vertex_normals[:, 0:3] = np.dot(vertex_normals[:, 0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)
            
            # Augment point cloud scale: 0.85x - 1.15x
            # Assertion: Augmenting scales of point clouds does not change normals
            scale_ratio = np.random.random() * 0.3 + 0.85
            point_cloud[:, 0:3] *= scale_ratio
            target_bboxes[:, 0:3] *= scale_ratio
            target_bboxes[:, 3:6] *= scale_ratio

        
        ret_dict = {
            # Basic
            "scan_name": scan_name,
            "point_clouds": point_cloud.astype(np.float32),
            "vertex_normals": selected_vertex_normals.astype(np.float32),
            # Data augmentation
            "ema_point_clouds": ema_point_clouds.astype(np.float32),
            "flip_x_axis": np.array(flip_YZ_XZ)[..., 0].astype(np.int64),
            "flip_y_axis": np.array(flip_YZ_XZ)[..., 1].astype(np.int64),
            "rot_mat": rot_mat.astype(np.float32),
            "scale": np.array(scale_ratio).astype(np.float32),
            # Label
            "center_label": target_bboxes.astype(np.float32)[:,0:3],
            "heading_class_label": angle_classes.astype(np.int64),
            "heading_residual_label": angle_residuals.astype(np.float32),
            "size_label": target_bboxes.astype(np.float32)[:,3:6],
            "num_gt_boxes": num_gt_boxes.astype(np.int64)
        }

        if self.split_set == "valid":

            target_quad_centers = np.zeros((MAX_NUM_QUAD,3)) 
            target_normal_vectors = np.zeros((MAX_NUM_QUAD,3)) 
            target_quad_sizes = np.zeros((MAX_NUM_QUAD,2))  

            mapping_name = self.valid_mapping[scan_name]
            z = point_cloud[..., -1]
            height_A = np.percentile(z, 98) 
            height_B = np.percentile(z, 5)
            center_z = (height_A + height_B) / 2
            height = height_A - height_B
            rectangles = arkitscenes_utils.get_quads(mapping_name, height=height, center_z=center_z)

            if rectangles.shape[0]>0:                                                
                target_quad_centers[0:rectangles.shape[0],:]=rectangles[:,0:3]         
                target_normal_vectors[0:rectangles.shape[0],:]=rectangles[:,3:6]
                target_quad_sizes[0:rectangles.shape[0],:]=rectangles[:,6:8]

            ret_dict['gt_quad_centers'] = target_quad_centers.astype(np.float32)
            ret_dict['gt_quad_sizes'] = target_quad_sizes.astype(np.float32)
            ret_dict['gt_normal_vectors'] = target_normal_vectors.astype(np.float32)
            
            num_gt_quads = np.zeros((NUM_QUAD_PROPOSAL))+ rectangles.shape[0]
            ret_dict['num_gt_quads'] =  num_gt_quads.astype(np.int64)
            num_total_quads = np.zeros((NUM_QUAD_PROPOSAL)) + rectangles.shape[0]
            ret_dict['num_total_quads'] =  num_total_quads.astype(np.int64)
            
            target_horizontal_quads = np.zeros((4,4,3))                                   
            ret_dict['horizontal_quads'] = target_horizontal_quads.astype(np.float32)

        return ret_dict


if __name__ == "__main__":
    dset = ARKitSceneDataset(split_set="train")
    from tqdm import tqdm
    for example in tqdm(dset):
        pc = example['point_clouds']
        normal = example['vertex_normals']
        # center = example['center_label']
        # size = example['size_label']
        # scan_name = example['scan_name']
        # box = np.concatenate([center, size], axis=1)
        # os.makedirs("../dump/ARKitDump/", exist_ok=True)
        # dump_pc(pc, f"../dump/ARKitDump/{scan_name}_pc.txt", normal)
        # pc_util.write_bbox(box, f"../dump/ARKitDump/{scan_name}_box.ply")
        # print(pc.shape[0])
        # assert pc.shape[0] > 40000, f"{example['scan_name']}"
