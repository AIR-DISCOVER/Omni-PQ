import os
import sys
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from utils import pc_util

def get_quads(mapping_name, height=2.00, center_z = 1.00):
    with open(os.path.join(BASE_DIR, "data", "annotations", f"{mapping_name}.json")) as f:
        text = f.read().strip()
    js = json.loads(text)
    data = js['labels']
    
    box_num = len(data)
    if box_num > 0:
        center = np.stack([
            np.stack([
                box['box3d']['location']['x'],
                box['box3d']['location']['y'],
                box['box3d']['location']['z'],
            ])
            for box in data])
        
        center[..., 2] = center_z
        
        dxyz = np.stack([
            np.stack([
                box['box3d']['dimension']['width'],
                box['box3d']['dimension']['length'],
                box['box3d']['dimension']['height'],
            ])
            for box in data])
        
        width = np.max(dxyz[..., :2], axis=1)
        normal_dir = np.argmin(dxyz[..., :2], axis=1)[..., None]
        
        normal = np.repeat(np.array([[1, 0, 0]]), (box_num, ), axis=0) * (1-normal_dir) +\
            np.repeat(np.array([[0, 1, 0]]), (box_num, ), axis=0) * normal_dir
        
        size = np.stack([width, np.repeat(height, box_num)], axis=1)
        
        rectangle = np.concatenate((center, normal, size), axis=1)
        return rectangle

    else:
        return np.zeros(shape=(0, 8))

if __name__ == "__main__":
    rectangle = get_quads("000000")
    