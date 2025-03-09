import os
import torch
import clip
import open3d as o3d
import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Union
from vlmaps.utils.matterport3d_categories import mp3dcat,mp3dcat_s
from vlmaps.utils.clip_utils import get_text_feats
from vlmaps.utils.mapping_utils import load_3d_map

def read_3d_map(map_path: str) -> Tuple[Set[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load 3D voxel map with features

    Args:
        map_path (str): path to save the map as an H5DF file.
    Return:
        mapped_iter_list (Set[int]): stores already processed frame's number.
        grid_feat (np.ndarray): (N, feat_dim) features of each 3D point.
        grid_pos (np.ndarray): (N, 3) each row is the (row, col, height) of an occupied cell.
        weight (np.ndarray): (N,) accumulated weight of the cell's features.
        occupied_ids (np.ndarray): (gs, gs, vh) either -1 or 1. 1 indicates
            occupation.
        grid_rgb (np.ndarray, optional): (N, 3) each row stores the rgb value
            of the cell.
        ---
        N is the total number of occupied cells in the 3D voxel map.
        gs is the grid size (number of cells on each side).
        vh is the number of cells in height direction.
    """
    with h5py.File(map_path, "r") as f:
        mapped_iter_list = f["mapped_iter_list"][:].tolist()# len():441 0-400
        grid_feat = f["grid_feat"][:]# shape:(67929, 512)
        grid_pos = f["grid_pos"][:] # shape:(67929, 3)
        weight = f["weight"][:]# shape:(67929,)
        occupied_ids = f["occupied_ids"][:]# shape:(1000, 1000, 30)
        grid_rgb = None
        mask_array = None
        if "grid_rgb" in f:
            grid_rgb = f["grid_rgb"][:]# shape:(67929, 3)
        if "mask_array" in f:
            mask_array = f["mask_array"][:]
            
    print('mapped_iter_list',len(mapped_iter_list),'\n')
    print('grid_feat',grid_feat.shape,'\n')
    print('grid_pos',grid_pos.shape,'\n')
    print('weight',weight.shape,'\n')
    print('occupied_ids',occupied_ids.shape,'\n')
    if grid_rgb is not None:
        print('grid_rgb',grid_rgb.shape,'\n')
    if mask_array is not None:
        print('mask_array',mask_array.shape,'\n')
    return mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb,mask_array

def visual_pcd (coord_instance,label_instance):
    # Get the unique labels in seg_data
    unique_labels = np.unique(label_instance)#shape=(74,)  min=-1,max=73

    # Generate random colors for each label
    color_map = {label: np.random.rand(3) for label in unique_labels}# len=74

    # Create an array for colors using the color_map
    colors = np.array([color_map[label] for label in label_instance]) #(232453, 3)



    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()
    # Assign coordinates to the PointCloud object
    pcd.points = o3d.utility.Vector3dVector(coord_instance)
    # Assign colors to the PointCloud object
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals()
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.59999999999999987,
                                    front=[ -0.032748670994499005, 0.5200064740314182, 0.85353429428084859 ],
                                    lookat=[ 448.50340458228868, 506.31944289998978, 37.421089203841127 ],
                                    up=[ 0.010820475881906882, -0.85375784568963453, 0.52055783369869968 ],)

                                    
    return 0
# Define the file paths
pcd_seg_filepath = '/home/ztl/deeplearning/vlmaps/vlmaps_dataset/vlmaps_dataset/jh4fc5c5qoQ_3/vlmap/vlmaps_f.h5df'# Replace with your file path 3d_new_1024_f1.pth
map_save_path = '/home/ztl/deeplearning/vlmaps/vlmaps_dataset/vlmaps_dataset/jh4fc5c5qoQ_3/vlmap/vlmaps_f.h5df'
if __name__ == "__main__":
    # Load point cloud data
    (
    mapped_iter_list,
    grid_feat,
    grid_pos,
    weight,
    occupied_ids,
    grid_rgb,
    mask_array,
        ) = read_3d_map(pcd_seg_filepath)
    pred_masks = np.asarray((mask_array)).T 
    open_feature = np.zeros((pred_masks.shape[1],))

    for i, row in enumerate(pred_masks):
        # 检查x中的1的位置
        indices = np.where(row == 1)[0]
        # 对于每个索引，将y中的相应数据复制到z中
        for idx in indices:
            open_feature[idx] = i
    visual_pcd(grid_pos,open_feature)


'''
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 534.0, 624.0, 43.0 ],
			"boundingbox_min" : [ 354.0, 466.0, 8.0 ],
			"field_of_view" : 60.0,
			"front" : [ 0.14000872369242462, 0.5305400698902929, 0.83601721963774256 ],
			"lookat" : [ 451.68326191799503, 502.82483228983563, 51.002071841530565 ],
			"up" : [ -0.18330022470107815, -0.81585175128447418, 0.54844046855660711 ],
			"zoom" : 0.57999999999999985
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 534.0, 624.0, 43.0 ],
			"boundingbox_min" : [ 354.0, 466.0, 8.0 ],
			"field_of_view" : 60.0,
			"front" : [ -0.065590046782589381, 0.51887820175053057, 0.85232819823774075 ],
			"lookat" : [ 429.16006122885204, 503.37068416022856, 42.130606330782641 ],
			"up" : [ 0.014111150229752231, -0.85359977827474931, 0.52073822019176974 ],
			"zoom" : 0.59999999999999987
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
'''