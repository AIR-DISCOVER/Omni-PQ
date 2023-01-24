import os
import sys
import numpy as np
import json
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))


from models.dump_helper import dump_pc


def isFourPointsInSamePlane(p0, p1, p2, p3,error):
        s1 = p1-p0
        s2 = p2-p0
        s3 = p3-p0
        result = s1[0]*s2[1]*s3[2]+s1[1]*s2[2]*s3[0]+s1[2]*s2[0]*s3[1]-s1[2]*s2[1]*s3[0]-s1[0]*s2[2]*s3[1]-s1[1]*s2[0]*s3[2]
        if result - error <= 0 <= result + error:
            return True
        return False       


def get_normal(quad_vert,center):
    tmp_A = []
    tmp_b = []
    for i in range(4):
        tmp_A.append([quad_vert[i][0], quad_vert[i][1], 1]) #x,y,1
        tmp_b.append(quad_vert[i][2]) #z
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    temp=A.T * A
    if np.linalg.det(temp)>1e-10:
        fit = np.array(temp.I * A.T * b)
        a = fit[0][0]/fit[2][0]
        b = fit[1][0]/fit[2][0]
        c = -1.0/fit[2][0]
        normal_vector = np.array([a,b,c])
        
        #print ("solution:%f x + %f y + %f z + 1 = 0" % (a, b, c) )    
        
    else:  #vertical
        b=np.matrix([-1,-1,-1,-1]).T
        A=A[:,0:2]
        temp=A.T * A
        fit = np.array(temp.I * A.T * b)
        a=fit[0][0]
        b=fit[1][0]
        c=0
        normal_vector = np.array([a,b,c])
        #print ("solution:%f x + %f y + 1 = 0" % (a, b) )

    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    return normal_vector

def rectangle(quad_vert,center):
    """
    input: p1,p2,p3,p4
    return: normal vector, size, quad center, direction
    """

    quad_center=np.mean(quad_vert, axis=0) 

    normal_vector = get_normal(quad_vert,center)

    vertical_normal_vector = np.array([normal_vector[0],normal_vector[1],0])

    vertical_normal_vector = vertical_normal_vector/np.linalg.norm(vertical_normal_vector)
    
    edge_vector = quad_vert[0]-quad_vert[1]

    cos_theta = torch.cosine_similarity(torch.tensor(edge_vector),torch.tensor([0,0,1]),dim=0)    

    l1=np.linalg.norm(quad_vert[0]-quad_vert[1])
    l2=np.linalg.norm(quad_vert[1]-quad_vert[2])
    l3=np.linalg.norm(quad_vert[2]-quad_vert[3])
    l4=np.linalg.norm(quad_vert[3]-quad_vert[0])
    l5 = (l1+l3)/2
    l6 = (l2+l4)/2

    if abs(cos_theta) > 0.5:   
        h = np.array([l5])
        w = np.array([l6])
    else:
        h = np.array([l6])
        w = np.array([l5])

    rectangle = np.concatenate((quad_center,vertical_normal_vector,w,h))  #3+3+2=8
    

    return rectangle
    
def get_center(verts):
    verts = np.array(verts)
    center=np.mean(verts, axis=0) 
    return center

def transform(scan_name,mesh_vertices):
    meta_file = BASE_DIR + '/scans_transform/'+os.path.join(scan_name,scan_name+'.txt')
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:,0:3] = mesh_vertices[:,0:3]
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    mesh_vertices[:,0:3] = pts[:,0:3]
    return mesh_vertices

# For showing the ability of our methods, we use the same function as PQ-Transformer for training
def get_quads(scan_name):
    with open(BASE_DIR+'/scannet_planes/'+scan_name+'.json','r') as quad_file:
        plane_dict = json.load(quad_file)
    quad_dict = plane_dict['quads']
    total_quad_num = len(quad_dict)

    vert_dict=plane_dict['verts']
    
    for i in range(0,len(vert_dict)):
       temp = vert_dict[i][1]
       vert_dict[i][1] = - vert_dict[i][2]
       vert_dict[i][2] = temp

    verts = np.array(vert_dict)

    verts = transform(scan_name,verts)

    quads=[i for i in quad_dict if len(i)==4]  # Maybe buggy: This may ignore quads that has more than 4 vertices...

    quad_verts=np.asarray([[verts[j] for j in _] for _ in quads])


    quad_verts_filter_ = np.asarray([quad_vert for quad_vert in quad_verts 
                                        if isFourPointsInSamePlane(quad_vert[0],quad_vert[1],quad_vert[2],quad_vert[3],100)])


    room_center = get_center(vert_dict) #room center

    quad_verts_filter = np.asarray([quad_vert for quad_vert in quad_verts_filter_ 
                                        if abs(get_normal(quad_vert, room_center)[2])<0.2]) #only vertical    
    
    horizontal_quads = np.asarray([quad_vert for quad_vert in quad_verts_filter_ 
                                        if abs(get_normal(quad_vert, room_center)[2])>0.8]) #only horizontal    
    

    rectangles = np.array([rectangle(_, room_center) for _ in quad_verts_filter])
    
    return rectangles,total_quad_num,horizontal_quads


# This is an improved version of get_quads, though not used in our paper
def get_quads_eval(scan_name):
    with open(BASE_DIR+'/scannet_planes/'+scan_name+'.json','r') as quad_file:
        plane_dict = json.load(quad_file)
    quad_dict = plane_dict['quads']
    total_quad_num = len(quad_dict)

    vert_dict = plane_dict['verts']
    room_center = get_center(vert_dict)
    
    for i in range(0,len(vert_dict)):
       temp = vert_dict[i][1]
       vert_dict[i][1] = - vert_dict[i][2]
       vert_dict[i][2] = temp
    
    verts = np.array(vert_dict)
    verts = transform(scan_name, verts)
    
    rectangles, horizontal_quads = [], []
    for quad_iid, quad in enumerate(quad_dict):
        # We estimate a normal vector for this quad
        quad_vertices = [verts[iid] for iid in quad]
        
        # First, we assert that all points are in the same plane
        for i in range(3, len(quad_vertices)):
            assertion = isFourPointsInSamePlane(quad_vertices[0], quad_vertices[1], quad_vertices[2], quad_vertices[i], 10)
            if not assertion:
                dump_pc(np.array(quad_vertices), "../dump/pc.txt", None)
            assert assertion,  f"{scan_name} {i} failed!"
        
        # Then, we estimate normal for this plane
        short_quad_verts = quad_vertices[:4]
        estimated_normal = get_normal(short_quad_verts, room_center)
        if abs(estimated_normal[2]) > 0.8:
            if len(quad) == 4:
                horizontal_quads.append(quad_vertices)
        
        elif abs(estimated_normal[2]) < 0.2:
            quad_vertices = np.array(quad_vertices)  # N x 3
            mean_z = np.mean(quad_vertices, axis=0)[2]
            # Divide into upper part and lower part
            upper_part_mask = quad_vertices[..., 2] > mean_z
            upper_quad_vertices = quad_vertices[upper_part_mask, ...]
            lower_quad_vertices = quad_vertices[~upper_part_mask, ...]
            
            axis_dir = np.cross(np.array([0, 0, 1]), np.array([estimated_normal[0], estimated_normal[1], 0]))
            axis_dir = axis_dir / np.linalg.norm(axis_dir)
            
            vert_reconstructed = []
            
            # upper
            upper_inner_dot = np.dot(upper_quad_vertices, axis_dir)
            ind1, ind2 = np.argmax(upper_inner_dot), np.argmin(upper_inner_dot)
            vert_reconstructed.append(upper_quad_vertices[ind1])
            vert_reconstructed.append(upper_quad_vertices[ind2])
            
            # lower
            lower_inner_dot = np.dot(lower_quad_vertices, axis_dir)
            ind1, ind2 = np.argmax(lower_inner_dot), np.argmin(lower_inner_dot)
            vert_reconstructed.append(lower_quad_vertices[ind2])
            vert_reconstructed.append(lower_quad_vertices[ind1])
            
            vert_reconstructed = np.array(vert_reconstructed)
            rectangles.append(rectangle(vert_reconstructed, room_center))
            
    rectangles = np.array(rectangles)
    horizontal_quads = np.array(horizontal_quads)

    return rectangles, total_quad_num, horizontal_quads

if __name__ == "__main__":
    from scannet.scannet_detection_dataset import ScannetDetectionDataset
    from tqdm import tqdm
    dset = ScannetDetectionDataset(split_set="val", num_points=40000)
    for scan in tqdm(dset):
        print("scan")