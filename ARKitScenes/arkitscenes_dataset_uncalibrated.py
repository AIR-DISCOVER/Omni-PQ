import os
import random
import sys

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

import trimesh

def convert_oriented_box_to_trimesh_fmt(box, color=None):
        
        def heading2rotmat(heading_angle):
            rotmat = np.zeros((3,3))
            rotmat[2,2] = 1
            cosval = np.cos(heading_angle)
            sinval = np.sin(heading_angle)
            rotmat[0:2,0:2] = np.array([[cosval, -sinval],[sinval, cosval]])
            return rotmat
        
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3,3] = 1.0            
        trns[0:3,0:3] = heading2rotmat(box[6])
        
        if color is None:
            color = (1, 1, 1, 0.3)
        
        box_trimesh_fmt = trimesh.creation.box(lengths, trns, visual=trimesh.visual.ColorVisuals(
            face_colors=list((color for k in range(12)))
        ))
        return box_trimesh_fmt    

def write_oriented_bbox(scene_bbox, out_filename, colors=None, single=False):
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename: (string) filename
        colors: a 4-tuple (rgbd)
    """
    scene = trimesh.scene.Scene()
    
    if len(scene_bbox) > 0:
                
        zeros = np.zeros(shape=(scene_bbox.shape[0], 1))
        scene_bbox = np.concatenate([scene_bbox, zeros], axis=1)
        
        if colors is None:
            colors = []
            for i in range(len(scene_bbox)):
                colors.append((0.7, 0.7, 0.7, 0.3))
                

        for box, color in zip(scene_bbox, colors):
            scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box, color=color))
        
        mesh_list = trimesh.util.concatenate(scene.dump())
        # save to ply file    
        mesh_list.export(out_filename, file_type='ply')

    else:
        with open(out_filename, "w+") as f:
            f.write("")
    
    return


def dump_pc_colored(point_clouds, dump_name="./dump/tmp.txt", colors=None):
    OUT = ""
    with open(dump_name, "w+") as f:
        if colors is None:
            for i in range(point_clouds.shape[0]):
                OUT += f"{point_clouds[i][0]} {point_clouds[i][1]} {point_clouds[i][2]} 0.0 0.0 0.0 0.2 0.2 0.2 1.0\n"

        else:
            for i in range(point_clouds.shape[0]):
                OUT += f"{point_clouds[i][0]} {point_clouds[i][1]} {point_clouds[i][2]} 0.0 0.0 0.0 {colors[i][0]} {colors[i][1]} {colors[i][2]} 1.0\n"

        f.write(OUT)


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

        self.scan_names = sorted(split_filenames)
        bak_scan_names = self.scan_names
        
        self.start_idx = int(len(self.scan_names) * start_proportion)
        self.end_idx = int(len(self.scan_names) * end_proportion)
        self.scan_names = self.scan_names[self.start_idx:self.end_idx]
    
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
        mesh_vertices = np.load(os.path.join(scan_dir, f"{scan_name}_data", f"{scan_name}_pc.npy"))
        instance_bboxes = np.load(os.path.join(scan_dir, f"{scan_name}_label", f"{scan_name}_bbox.npy"), allow_pickle=True).item()

        # Prepare label containers
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
            
        
        bbox_num = min(instance_bboxes['bboxes'].shape[0], MAX_NUM_OBJ)
        target_bboxes[0:bbox_num, :] = instance_bboxes['bboxes'][:, 0:6]
        
        target_bboxes_mask[0:bbox_num] = 1
        bbox_cls = instance_bboxes['types']
        num_gt_boxes = np.zeros((NUM_PROPOSAL)) + bbox_num

        # TODO: downsample points
        point_cloud, choices = pc_util.random_sampling(mesh_vertices,
            self.num_points, return_choices=True)    
        
        ema_point_clouds, ema_choices = pc_util.random_sampling(mesh_vertices,
            self.num_points, return_choices=True)

        
        # TODO: OBB-Guided Scene Axis-Alignment: Rotation
        angle = np.percentile(instance_bboxes['bboxes'][..., -1] % (np.pi / 2), 50)
        rot_mat = pc_util.rotz(angle)
        point_cloud[:, 0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
        ema_point_clouds[:, 0:3] = np.dot(ema_point_clouds[:, 0:3], np.transpose(rot_mat))
        target_bboxes[:, 0:3] = np.dot(target_bboxes[:, 0:3], np.transpose(rot_mat))
        
        former_angle = instance_bboxes['bboxes'][..., -1]
        former_angle -= angle
        former_angle %= 2 * np.pi
        reverse_mask = ((np.pi / 4 <= former_angle) & (former_angle <= np.pi / 4 * 3)) | \
            ((np.pi / 4 * 5 <= former_angle) & (former_angle <= np.pi / 4 * 7))
        padding = np.zeros((target_bboxes.shape[0]-reverse_mask.shape[0]))
        reverse_mask = np.concatenate([reverse_mask, padding], axis=0)
        dx = np.copy(target_bboxes[..., 3])
        dy = np.copy(target_bboxes[..., 4])
        target_bboxes[..., 3] = dy * reverse_mask + dx * (1-reverse_mask)
        target_bboxes[..., 4] = dx * reverse_mask + dy * (1-reverse_mask)
        
        # TODO: OBB-Guided Scene Axis-Alignment: Translation
        z_filter_L = np.percentile(point_cloud[..., 2], 15)
        z_filter_H = np.percentile(point_cloud[..., 2], 85)
        filter_mask = (point_cloud[..., 2] >= z_filter_L) & (point_cloud[..., 2] <= z_filter_H)
        x_base = np.percentile(point_cloud[filter_mask, 0], 50)
        y_base = np.percentile(point_cloud[filter_mask, 1], 50)
        z_base = np.percentile(point_cloud[..., 2], 5)
        offset = np.array([x_base, y_base, z_base])
    
        point_cloud[:, 0:3] = point_cloud[:, 0:3] - offset[None, ...]
        ema_point_clouds[:, 0:3] = ema_point_clouds[:, 0:3] - offset[None, ...]
        target_bboxes[:, 0:3] = target_bboxes[:, 0:3] - offset[None, ...]

        
        # TODO: data augmentation
        flip_YZ_XZ = np.array([False, False])
        rot_angle = np.array(0.0)
        scale_ratio = np.array(1.0)
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                flip_YZ_XZ[0] = True
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                target_bboxes[:, 0] = -1 * target_bboxes[:, 0]
            
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                flip_YZ_XZ[0] = False
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                target_bboxes[:, 1] = -1 * target_bboxes[:, 1]
            
            # Rotation along up-axis / Z-axis
            # rot_angle = (np.random.random()*np.pi/18) - np.pi/36
            rot_angle = 0.0
            rot_angle += random.choice([0, 1, 2, 3]) * np.pi / 2
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            from scannet.model_util_scannet import rotate_aligned_boxes,rotate_quad
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
            # Data augmentation
            "ema_point_clouds": ema_point_clouds.astype(np.float32),
            "flip_x_axis": np.array(flip_YZ_XZ)[..., 0].astype(np.int64),
            "flip_y_axis": np.array(flip_YZ_XZ)[..., 1].astype(np.int64),
            "rot_angle": np.array(rot_angle).astype(np.float32),
            "scale": np.array(scale_ratio).astype(np.float32),
            # Label
            "boxes": target_bboxes.astype(np.float32),
            "num_gt_boxes": num_gt_boxes.astype(np.int64)
        }


        return ret_dict


if __name__ == "__main__":
    dset = ARKitSceneDataset(split_set="train", augment=True)
    from tqdm import tqdm
    i=0
    for example in tqdm(dset):
        print(example['scan_name'])
        dump_pc(example['point_clouds'], f"./show/{example['scan_name']}.txt")
        write_oriented_bbox(example['boxes'], f"./show/{example['scan_name']}.ply")
        print(example['rot_angle'])
        if i == 5:
            break
        i+=1
        # center = example['center_label']
        # size = example['size_label']
        # scan_name = example['scan_name']
        # box = np.concatenate([center, size], axis=1)
        # os.makedirs("../dump/ARKitDump/", exist_ok=True)
        # dump_pc(pc, f"../dump/ARKitDump/{scan_name}_pc.txt", normal)
        # pc_util.write_bbox(box, f"../dump/ARKitDump/{scan_name}_box.ply")
        # print(pc.shape[0])
        # assert pc.shape[0] > 40000, f"{example['scan_name']}"
