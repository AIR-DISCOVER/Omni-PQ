import os
from tqdm import tqdm
import numpy as np
import pymeshlab


def process():
    scene_files = os.listdir("./scannet_train_detection_data/")
    vert_files = sorted([x for x in scene_files if "vert.npy" in x])  # file_name + "_vert.npy"
    file_names = [x[:len("scene0000_00")] for x in vert_files]
    all_scan_names = list(set([os.path.basename(x)[0:12] \
            for x in os.listdir("./scannet_planes") if x.startswith('scene')]))
    split_filenames = 'meta_data/scannetv2_train.txt'
    with open(split_filenames, 'r') as f:
        scan_names = f.read().splitlines()

    # remove unavailiable scans
    scan_names = sorted([sname for sname in scan_names if sname in file_names and sname in all_scan_names])
    
    os.makedirs("scannet_train_detection_data_normals", exist_ok=True)
    
    for file_idx, filename in tqdm(enumerate(scan_names), total=len(scan_names)):

        tqdm.write("Start processing " + filename)
        
        vert_data = np.load(f"./scannet_train_detection_data/{filename}_vert.npy")  # (50000, 6)
        sem_label_data = np.load(f"./scannet_train_detection_data/{filename}_sem_label.npy")  # (50000, )
        
        ms = pymeshlab.MeshSet()
        
        with open("buffer.TXT", 'w+') as f:
            # print(data[:2, :])
            layout_pt_cnt = vert_data.shape[0]
            pc_center = np.mean(vert_data, axis=0)[:3]
            pc_center[2] = (np.max(vert_data, axis=0)[2] + pc_center[2]) / 2
            f.write("\n".join(
                [f"{x[0]} {x[1]} {x[2]}" for x in vert_data]
            ))
        ms.load_new_mesh("buffer.TXT", separator=2)  # Too shame, it could only pass in a `filename`
        ms.apply_filter('compute_normals_for_point_sets', k=100, smoothiter=5, flipflag=True, viewpos=pc_center)
        
        normal_vectors = ms.current_mesh().vertex_normal_matrix()

        reverse_mask = ((vert_data[:, :3] - pc_center[:3]).reshape(layout_pt_cnt, 1, 3) \
            @ normal_vectors.reshape(layout_pt_cnt, 3, 1)).reshape(layout_pt_cnt) < 0
            
        normal_vectors[reverse_mask] = -normal_vectors[reverse_mask]  # Point towards inner
        np.save(f"./scannet_train_detection_data_normals/{filename}.normal.npy", normal_vectors)


if __name__ == "__main__":
    process()
