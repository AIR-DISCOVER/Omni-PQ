import os, sys
import numpy as np
import open3d as o3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
sys.path.append(os.path.join(ROOT_DIR))
data_path = os.path.join(BASE_DIR, 'dataset', "3dod/Training")

from utils import pc_util
from models.dump_helper import dump_pc


def save_transformed_scene(scan_name):
    scan_dir = os.path.join(data_path, scan_name, f"{scan_name}_offline_prepared_data")    
    mesh_file = os.path.join(data_path, scan_name, f"{scan_name}_3dod_mesh.ply")
    
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    
    # Load scene axis alignment matrix
    instance_bboxes = np.load(os.path.join(scan_dir, f"{scan_name}_bbox.npy"), allow_pickle=True).item()
    angle = np.percentile(instance_bboxes['bboxes'][..., -1] % (np.pi / 2), 50)
    rot_mat = pc_util.rotz(angle)
    
    t_matrix_1 = np.zeros(shape=(4, 4))
    t_matrix_1[3, 3] = 1.
    t_matrix_1[:3, :3] = rot_mat
    mesh1 = mesh.transform(t_matrix_1)
    
    mesh_vertices_prime = np.asarray(mesh1.vertices)
    z_filter_L = np.percentile(mesh_vertices_prime[..., 2], 15)
    z_filter_H = np.percentile(mesh_vertices_prime[..., 2], 85)
    filter_mask = (mesh_vertices_prime[..., 2] >= z_filter_L) & (mesh_vertices_prime[..., 2] <= z_filter_H)
    x_base = np.percentile(mesh_vertices_prime[filter_mask, 0], 50)
    y_base = np.percentile(mesh_vertices_prime[filter_mask, 1], 50)
    z_base = np.percentile(mesh_vertices_prime[..., 2], 5)
    offset = -np.array([x_base, y_base, z_base])
    
    t_matrix_2 = np.eye(4)
    t_matrix_2[3, 3] = 1.
    t_matrix_2[:3, 3] = offset
    mesh2 = mesh1.transform(t_matrix_2)
    
    mesh3 = mesh2.simplify_vertex_clustering(0.032)
    vertices = np.asarray(mesh3.vertices)
    delete_thres = np.percentile(vertices[..., 2], 80)
    filter_mask = (vertices[..., 2] >= delete_thres)
    mesh3.remove_vertices_by_mask(filter_mask)
    
    save_dir = os.path.join(data_path, scan_name, f"{scan_name}_offline_prepared_data", f"{scan_name}_3dod_mesh_transformed.ply")
    o3d.io.write_triangle_mesh(save_dir, mesh3)
    
    pc = np.asarray(mesh3.vertices)
    save_dir = os.path.join(data_path, scan_name, f"{scan_name}_offline_prepared_data", f"{scan_name}_pc.npy")
    np.save(save_dir, pc)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))
    normal = np.asarray(pcd.normals)
    save_dir = os.path.join(data_path, scan_name, f"{scan_name}_offline_prepared_data", f"{scan_name}_normal.npy")
    np.save(save_dir, normal)


if __name__ == "__main__":
    
    all_scan_name = open("./dataset/train_filtered.txt").read().strip().split("\n")
    from p_tqdm import p_map
    p_map(save_transformed_scene, all_scan_name, num_cpus=128)
