import os
import torch
import clip
import open3d as o3d
import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Union
from vlmaps.utils.matterport3d_categories import mp3dcat,mp3dcat_s,ithor_obj,ithor_s
from vlmaps.utils.clip_utils import get_text_feats
from vlmaps.utils.mapping_utils import load_3d_map
import time
def clip_feature_extract(grid_feat,grid_pos,visual,save_pcd):


    # seg_data = torch.load(pcd_seg_filepath)#shape=(232453,),min=-1,max=72 

    # Get the coordinates
    # coordinates1 = seg_data['coord'].astype('float64') 
    # coordinates2 = np.around(coordinates1,decimals=2)#.astype(np.int32)
    # coordinates = np.array([np.array(row) for row in coordinates2])
    coordinates = grid_pos #seg_data['coord'].astype('float64')  # convert to float64 for o3d /shape=(232453, 3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    clip_version = "ViT-B/32"
    clip_feat_dim = {
        "RN50": 1024,
        "RN101": 512,
        "RN50x4": 640,
        "RN50x16": 768,
        "RN50x64": 1024,
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
        }[clip_version]
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(clip_version,device=device)  # clip.available_models()
    clip_model.to(device).eval()
    lang = ithor_obj#mp3dcat_#sithor_s
    text_feats = get_text_feats(lang, clip_model, clip_feat_dim)
    # text_feats = np.concatenate((text_feats,text_feats),1)

    data_s = grid_feat #seg_data['feat'] #important data !!!
    scores3d_list = data_s @ text_feats.T
    predicts3d = np.argmax(scores3d_list, axis=1)

    # Get the unique labels in seg_data
    unique_labels = np.unique(predicts3d)#shape=(74,)  min=-1,max=73

    # Generate random colors for each label
    color_map = {label: np.random.rand(3) for label in unique_labels}# len=74

    # Create an array for colors using the color_map
    colors = np.array([color_map[label] for label in predicts3d]) #(232453, 3)



    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()
    # Assign coordinates to the PointCloud object
    pcd.points = o3d.utility.Vector3dVector(coordinates)
    # Assign colors to the PointCloud object
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if save_pcd == True:
        assert 0
    if visual == True:
        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])
    return pcd,coordinates,data_s,predicts3d

def pc_separation (coord,feat,class_label,grad_rgb):
    # 创建一个字典来存储每个标签对应的x的行
    pc_dict = {label: {'coord': [], 'feat': [],'rgb': []} for label in set(class_label.flatten())}

    # 遍历x和y，根据y中的标签将x的行存储到对应的字典键中
    for i, label in enumerate(class_label):
        pc_dict[label]['coord'].append(coord[i])
        pc_dict[label]['feat' ].append(feat[i])
        pc_dict[label]['rgb' ].append(grad_rgb[i])

    # 将列表转换为numpy数组
    for label in pc_dict:
        pc_dict[label]['coord'] = np.array(pc_dict[label]['coord'])
        pc_dict[label]['feat'] = np.array(pc_dict[label]['feat'])
        pc_dict[label]['rgb'] = np.array(pc_dict[label]['rgb'])
    return pc_dict

def pc_statistical_outlier_removal(scene_pcd,visual=True):
    # pcd = o3d.io.read_point_cloud("desk.pcd")
    print("原始点云：", scene_pcd)

    # ------------------------- 统计滤波 --------------------------
    print("->正在进行统计滤波...")
    num_neighbors = 20 # K邻域点的个数
    std_ratio = 0.5 # 标准差乘数
    # 执行统计滤波，返回滤波后的点云sor_pcd和对应的索引ind
    sor_pcd, ind = scene_pcd.remove_statistical_outlier(num_neighbors, std_ratio)
    
    # ===========================================================
    if visual == True:
        sor_pcd.paint_uniform_color([0, 0, 1])
        print("统计滤波后的点云：", sor_pcd)
        sor_pcd.paint_uniform_color([0, 0, 1])
        # 提取噪声点云
        sor_noise_pcd = scene_pcd.select_by_index(ind,invert = True)
        print("噪声点云：", sor_noise_pcd)
        sor_noise_pcd.paint_uniform_color([1, 0, 0])
        # 可视化统计滤波后的点云和噪声点云
        o3d.visualization.draw_geometries([sor_pcd, sor_noise_pcd])

    return ind

def pc_radius_outlier_removal(scene_pcd,visual=False):
    print("原始点云：", scene_pcd)
    # ------------------------- 半径滤波 --------------------------
    print("->正在进行半径滤波...")
    num_points = 5  # 邻域球内的最少点数，低于该值的点为噪声点 35
    radius = 4    # 邻域半径大小 7.5
    # 执行半径滤波，返回滤波后的点云sor_pcd和对应的索引ind
    sor_pcd, ind = scene_pcd.remove_radius_outlier(num_points, radius)
    sor_pcd.paint_uniform_color([0, 0, 1])
    print("半径滤波后的点云：", sor_pcd)
    sor_pcd.paint_uniform_color([0, 0, 1])
    # 提取噪声点云
    sor_noise_pcd = scene_pcd.select_by_index(ind,invert = True)
    print("噪声点云：", sor_noise_pcd)
    sor_noise_pcd.paint_uniform_color([1, 0, 0])
    # ===========================================================
    if visual == True:
        # 可视化半径滤波后的点云和噪声点云
        o3d.visualization.draw_geometries([sor_pcd, sor_noise_pcd])
    return ind


def pc_cluster_dbscan(scene_pcd,visual=False):
    print("->正在加载点云... ")
    # pcd = o3d.io.read_point_cloud("test.pcd")
    print(scene_pcd)

    print("->正在DBSCAN聚类...")
    eps = 5#0.5          # 同一聚类中最大点间距 10
    min_points = 15#20     # 有效聚类的最小点数 20
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(scene_pcd.cluster_dbscan(eps, min_points, print_progress=True))
    max_label = labels.max()    # 获取聚类标签的最大值 [-1,0,1,2,...,max_label]，label = -1 为噪声，因此总聚类个数为 max_label + 1
    print(f"point cloud has {max_label + 1} clusters")
    if visual == True:
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0  # labels = -1 的簇为噪声，以黑色显示
        scene_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([scene_pcd])
    return labels

def pc_instance_operation(coord, feat,rgb):

    scene_pcd = o3d.geometry.PointCloud()

    # Assign coordinates to the PointCloud object
    scene_pcd.points = o3d.utility.Vector3dVector(coord)

    filtered_indexes = pc_radius_outlier_removal(scene_pcd)
    
    if len(filtered_indexes)<=20:
        print('[INFO] Not enough points!')
        return None,None,None,None
    coord_selected = np.array([coord[i] for i in filtered_indexes])
    feat_selected = np.array([feat[i] for i in filtered_indexes])
    rgb_selected = np.array([rgb[i] for i in filtered_indexes])
    scene_pcd.points = o3d.utility.Vector3dVector(coord_selected)
    label_instance = pc_cluster_dbscan(scene_pcd)
    # 创建一个布尔数组，其中label不等于-1
    keep_mask = (label_instance != -1)

    # 使用布尔数组来索引x1，只保留label不等于-1的行
    label_instance = label_instance[keep_mask]
    coord_selected = coord_selected[keep_mask]
    feat_selected = feat_selected[keep_mask]
    rgb_selected = rgb_selected[keep_mask]
    if len(label_instance)==0:
        print('[INFO] Not enough points after cluster!=================================')
        return None,None,None,None
    return coord_selected, feat_selected,label_instance,rgb_selected

def label_extend(label_instance):
    # 初始化新标签列表
    label_ext = []

    # 记录上一个标签的最大值
    last_max = 0

    # 遍历每个子数组
    for i,arr in enumerate (label_instance):
        if i ==0:
            unique_labels = np.unique(arr)
            # 创建从0开始的新标签
            new_labels = np.arange(len(unique_labels))

            # 创建标签映射字典
            label_map = dict(zip(unique_labels, new_labels))

            # 使用映射字典转换原始标签
            new_label = np.array([label_map[l] for l in arr])
            label_ext.append(new_label)
            last_max = len(unique_labels)
        else:
            offset = last_max-arr.min()
            new_label_ext = arr + offset
            unique_labels = np.unique(arr)
            last_max = len(unique_labels)+last_max
            label_ext.append(new_label_ext)
    label_ext = np.concatenate(label_ext)
    return label_ext

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
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
    return 0

def maks_list2mask_array(label_list):
    # # Define the file paths
    # pcd_seg_filepath = './data/5LpN3gDmAk7_1/map/instance_label_gg.pth' #3d_new_sam3d512_1.pth' # Replace with your file path

    # # Load point cloud data

    # seg_data = torch.load(pcd_seg_filepath)#shape=(232453,),min=-1,max=72 

    # Get the coordinates
    scores3d_list = label_list#seg_data['label']


    # 找出x中所有不同的数字
    unique_values = np.unique(scores3d_list)

    # 初始化一个空的mask数组
    mask = np.zeros((len(unique_values), len(scores3d_list)), dtype=bool)

    # 对于每个唯一的数字，创建一个布尔数组，表示x中的每个元素是否等于该数字
    for i, value in enumerate(unique_values):
        mask[i, np.in1d(scores3d_list, value)] = True

    # 打印结果
    # print(mask)
    print(mask.T.shape)
    # map_save_dir = os.path.join('data/5LpN3gDmAk7_1', "map")
    # map3df_save_path_new = os.path.join(map_save_dir, "mask_gg.pt")
    # torch.save(mask.T, map3df_save_path_new)
    
    return mask.T

def save_3d_map(
    save_path: str,
    grid_feat: np.ndarray,
    grid_pos: np.ndarray,
    weight: np.ndarray,
    occupied_ids: np.ndarray,
    mapped_iter_list: Set[int],
    grid_rgb: np.ndarray = None,
    mask_array: np.ndarray = None,
) -> None:
    """Save 3D voxel map with features

    Args:
        save_path (str): path to save the map as an H5DF file.
        grid_feat (np.ndarray): (N, feat_dim) features of each 3D point.
        grid_pos (np.ndarray): (N, 3) the position of the occupied cell.
        weight (np.ndarray): (N,) accumulated weight of the cell's features.
        occupied_ids (np.ndarray): (gs, gs, vh) either -1 or 1. 1 indicates
            occupation.
        mapped_iter_list (Set[int]): stores already processed frame's number.
        grid_rgb (np.ndarray, optional): (N, 3) each row stores the rgb value
            of the cell.
        ---
        N is the total number of occupied cells in the 3D voxel map.
        gs is the grid size (number of cells on each side).
        vh is the number of cells in height direction.
    """
    with h5py.File(save_path, "w") as f:
        f.create_dataset("mapped_iter_list", data=np.array(mapped_iter_list, dtype=np.int32))
        f.create_dataset("grid_feat", data=grid_feat)
        f.create_dataset("grid_pos", data=grid_pos)
        f.create_dataset("weight", data=weight)
        f.create_dataset("occupied_ids", data=occupied_ids)
        if grid_rgb is not None:
            f.create_dataset("grid_rgb", data=grid_rgb)
            f.create_dataset("mask_array", data=mask_array)
def pcd_instance_cluster(pcd,coordinates,feat,predicts3d,grid_rgb):
    coord_instance_all, feat_instance_all,label_instance_all,rgb_instance_all = [],[],[],[]
    pc_dict=pc_separation(coordinates,feat,predicts3d,grid_rgb)
    # 循环遍历labeled_data字典
    for label, data in pc_dict.items():
        x1_label = data['coord']
        x2_label = data['feat']
        rgb_label = data['rgb']
    
        
        print(f"Label: {label}")
        print(f"Size of coord: {x1_label.shape}")
        print(f"Size of feat: {x2_label.shape}")
        print(f"Size of rgb: {rgb_label.shape}")
    
        # 举例：对x1和x2进行某种操作（这里只是示例，具体操作根据实际需求来定）
        coord_instance, feat_instance,label_instance,rgb_instance = pc_instance_operation(x1_label, x2_label,rgb_label)
        # 检查返回值是否为None，如果是，则跳过添加到列表的操作
        if coord_instance is not None and feat_instance is not None and label_instance is not None and rgb_instance is not None:
            coord_instance_all.append(coord_instance)
            feat_instance_all.append(feat_instance)
            label_instance_all.append(label_instance)
            rgb_instance_all.append(rgb_instance)
        else:
            print(f"[INFO] Skipping label {label} due to insufficient points.")
    print(len(label_instance_all))
    label_instance = label_extend(label_instance_all)
    coord_instance_all = np.vstack(coord_instance_all)
    feat_instance_all =  np.vstack(feat_instance_all)
    rgb_instance_all =  np.vstack(rgb_instance_all)


    print(label_instance.shape)
    print(coord_instance_all.shape)
    print(feat_instance_all.shape)
    unique_labels = np.unique(label_instance)
    # 创建从0开始的新标签
    labels_len = np.arange(len(unique_labels))
    print(labels_len)

    visual_pcd(coord_instance_all,label_instance)

    maks_array = maks_list2mask_array(label_instance)

    # img_save_dir = 'data/5LpN3gDmAk7_1'
    # map_save_dir = os.path.join(img_save_dir, "map")
    # os.makedirs(map_save_dir, exist_ok=True)
    # instance_coord_feat = dict(coord=coord_instance_all, feat=feat_instance_all)
    # instance_coord_label = dict(coord=coord_instance_all, label=label_instance)
    # instance_feat_save_path = os.path.join(map_save_dir, f"instance_feat_gg.pth")
    # instance_label_save_path = os.path.join(map_save_dir, f"instance_label_gg.pth")
    # torch.save(instance_coord_feat, instance_feat_save_path)
    # torch.save(instance_coord_label, instance_label_save_path)
    return coord_instance_all,feat_instance_all,label_instance,maks_array,rgb_instance_all


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

# Define the file paths
pcd_seg_filepath = '/home/ztl/deeplearn/vlmaps_ithor/vlmaps_dataset/vlmaps_dataset/FloorPlan_Val2_2/vlmap/vlmaps.h5df'# Replace with your file path 3d_new_1024_f1.pth
map_save_path = '/home/ztl/deeplearn/vlmaps_ithor/vlmaps_dataset/vlmaps_dataset/FloorPlan_Val2_2/vlmap/vlmaps_ff.h5df'
if __name__ == "__main__":
    start = time.time()
    # Load point cloud data
    (
    mapped_iter_list,
    grid_feat,
    grid_pos,
    weight,
    occupied_ids,
    grid_rgb,
        ) = load_3d_map(pcd_seg_filepath)
    read_3d_map(pcd_seg_filepath)
    pcd,coordinates,feat,predicts3d = clip_feature_extract(grid_feat,grid_pos,visual = False,save_pcd=False)
    # map_save_path = pcd_seg_filepath
    coord_instance_all,feat_instance_all,label_instance,maks_array,grid_rgb_instance = pcd_instance_cluster(pcd,coordinates,feat,predicts3d,grid_rgb)
    save_3d_map(map_save_path, feat_instance_all, coord_instance_all, weight, occupied_ids, mapped_iter_list, grid_rgb_instance,maks_array)
    end =time.time()
    print(end-start)
    ########test_h5py_read########
    read_3d_map(map_save_path)
    ##############################