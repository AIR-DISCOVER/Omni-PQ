import os, sys
import numpy as np
import open3d as o3d
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'scannet'))

SCAN_PATH = os.path.join(ROOT_DIR, "scannet", "scans/scans/")


def save_transformed_scene(scan_name):
    # Load scene axis alignment matrix
    meta_file = os.path.join(SCAN_PATH, scan_name, scan_name + '.txt')
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))

    # Rotate scene
    mesh_file = os.path.join(SCAN_PATH, scan_name, scan_name + '_vh_clean_2.ply')
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh3 = mesh.transform(axis_align_matrix)

    save_mesh_file = os.path.join(SCAN_PATH, scan_name, scan_name + '_transformed.ply')
    o3d.io.write_triangle_mesh(save_mesh_file, mesh3)


if __name__ == "__main__":
    scan_name = "scene0626_02"
    save_transformed_scene(scan_name=scan_name)
