import io
from math import ceil
import os
import time
from tqdm import tqdm
from p_tqdm import p_map
import numpy as np
import pymeshlab


def calc_normal(scene_name, prefix="Training"):
    
    save_dir = os.path.join(f"./3dod/{prefix}/{scene_name}/{scene_name}_offline_prepared_data/{scene_name}_data", f"{scene_name}_normal.npy")
    try:
        N = np.load(save_dir)
        return
    except:

        vert_data = np.load(os.path.join(f"./3dod/{prefix}/{scene_name}/{scene_name}_offline_prepared_data/{scene_name}_data", f"{scene_name}_pc.npy"))  # (???, 6)
        
        ms = pymeshlab.MeshSet()
        
        with open(f"buffer_{scene_name}.TXT", 'w+') as f:
            layout_pt_cnt = vert_data.shape[0]
            pc_center = np.mean(vert_data, axis=0)[:3]
            pc_center[2] = (np.max(vert_data, axis=0)[2] + pc_center[2]) / 2
            f.write("\n".join(
                [f"{x[0]} {x[1]} {x[2]}" for x in vert_data]
            ))
        ms.load_new_mesh(f"buffer_{scene_name}.TXT", separator=2)  # Too shame, it could only pass in a `filename`
        ms.apply_filter('compute_normals_for_point_sets', k=100, smoothiter=5, flipflag=True, viewpos=pc_center)
        
        normal_vectors = ms.current_mesh().vertex_normal_matrix()

        reverse_mask = ((vert_data[:, :3] - pc_center[:3]).reshape(layout_pt_cnt, 1, 3) \
            @ normal_vectors.reshape(layout_pt_cnt, 3, 1)).reshape(layout_pt_cnt) < 0
            
        normal_vectors[reverse_mask] = -normal_vectors[reverse_mask]  # Point towards inner
        np.save(save_dir, normal_vectors)
        
        os.remove(f"buffer_{scene_name}.TXT")
        del ms
    

def process():

    with open("train_filtered.txt") as f:
        scene_names = f.read().strip().split("\n")
    
    p_map(calc_normal, scene_names, ["Training"] * len(scene_names), num_cpus=16)
    
    with open("valid_filtered.txt") as f:
        scene_names = f.read().strip().split("\n")
    
    p_map(calc_normal, scene_names, ["Validation"] * len(scene_names), num_cpus=16)
    

if __name__ == "__main__":
    process()
