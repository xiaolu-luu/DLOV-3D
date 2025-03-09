
import clip
import numpy as np
import imageio
import torch
from tqdm import tqdm
import os
from pathlib import Path
from openmask3d.data.load import Camera, InstanceMasks3D, Images, PointCloud, get_number_of_images
from openmask3d.mask_features_computation.utils import initialize_sam_model, mask2box_multi_level, run_sam

class PointProjector:
    def __init__(self, camera: Camera,    #相机参数
                 point_cloud: PointCloud, #点云数据
                 masks: InstanceMasks3D,  #实例mask
                 vis_threshold,           #可视化阈值
                 indices):                #视角索引
        self.vis_threshold = vis_threshold
        self.indices = indices
        self.camera = camera
        self.point_cloud = point_cloud
        self.masks = masks
        self.visible_points_in_view_in_mask, self.visible_points_view, self.projected_points, self.resolution = self.get_visible_points_in_view_in_mask()
        self.point_coords=np.zeros((2,2))
        
    def get_visible_points_view(self):
        '''
        获取可见点和投影点方法 (get_visible_points_view):

        从相机中加载位姿信息、点云数据和深度图。
        计算投影点的坐标，并进行归一化处理。
        计算点云在每个视角的可见性，以确定哪些点在图像中可见。
        返回可见点、投影点和图像分辨率。
        '''
        # Initialization
        vis_threshold = self.vis_threshold #可视化阈值
        indices = self.indices             #视角索引
        depth_scale = self.camera.depth_scale #深度信息
        poses = self.camera.load_poses(indices) # 位姿 shape:(238, 3, 4)
        X = self.point_cloud.get_homogeneous_coordinates() # 获取点云 X.shape = (237360,4)
        n_points = self.point_cloud.num_points #点云个数
        depths_path = self.camera.depths_path  #深度图路经
        data_dir = Path(depths_path)
        depth_paths = sorted(data_dir.glob("*.npy"))

        resolution = np.load(os.path.join(depths_path, '000000'+ '.npy')).shape
        # resolution = imageio.imread(os.path.join(depths_path, '0.png')).shape #获得深度图分辨率resolution=(480,640)
        height = resolution[0]
        width = resolution[1]
        intrinsic = self.camera.get_adapted_intrinsic(resolution) #获得相机内参
        
        projected_points = np.zeros((len(indices), n_points, 2), dtype = int)#shape:(238, 237360, 2)
        visible_points_view = np.zeros((len(indices), n_points), dtype = bool)#shape:(238, 237360)
        print(f"[INFO] Computing the visible points in each view.")
        
        for i, idx in tqdm(enumerate(indices)): # for each view i=0~237 idx=0,10,20~2370
            # *******************************************************************************************************************
            # STEP 1: get the projected points 得到投影点
            # Get the coordinates of the projected points in the i-th view (i.e. the view with index idx)
            # 获取第i个视图(即索引为idx的视图)中投影点的坐标
            projected_points_not_norm = (intrinsic @ poses[i] @ X.T).T #projected_points_not_norm.shape = (237360, 3)
            # Get the mask of the points which have a non-null third coordinate to avoid division by zero
            # 获取具有非空第三坐标的点的掩码，以避免被零除 # mask.shape = (237360,) min = max = true (always)
            mask = (projected_points_not_norm[:, 2] != 0) # don't do the division for point with the third coord equal to zero 不要在第三个数等于0的情况下做点的除法
            # Get non homogeneous coordinates of valid points (2D in the image) 得到有效点的非齐次坐标(图像中的2D)
            projected_points[i][mask] = np.column_stack([[projected_points_not_norm[:, 0][mask]/projected_points_not_norm[:, 2][mask], 
                    projected_points_not_norm[:, 1][mask]/projected_points_not_norm[:, 2][mask]]]).T
            
            # *******************************************************************************************************************
            # STEP 2: occlusions computation 遮挡计算
            # Load the depth from the sensor
            # depth_path = os.path.join(depths_path, '5LpN3gDmAk7_'+ str(idx+1) + '.npy') 
            #sensor_depth = imageio.imread(depth_path) / depth_scale#(480, 640)
            sensor_depth = np.load(depth_paths[idx])
            inside_mask = (projected_points[i,:,0] >= 0) * (projected_points[i,:,1] >= 0) \
                                * (projected_points[i,:,0] < width) \
                                * (projected_points[i,:,1] < height) #inside_mask.shape = (237360,) [false , true]
            pi = projected_points[i].T # pi.shape=(2, 237360) max:36937949 min:-72308046
            # Depth of the points of the pointcloud, projected in the i-th view, computed using the projection matrices
            # 点云的点的深度，投影在第i个视图中，使用投影矩阵计算
            point_depth = projected_points_not_norm[:,2] # max:5.617117573936433 min:-1.0108500549511596 shape:(237360,)
            # Compute the visibility mask, true for all the points which are visible from the i-th view
            # 计算可见性掩码，对于从第i个视图可见的所有点为true
            visibility_mask = (np.abs(sensor_depth[pi[1][inside_mask], pi[0][inside_mask]]
                                        - point_depth[inside_mask]) <= \
                                        vis_threshold).astype(bool) #visibility_mask.shape=(68019,) [false,true]
            inside_mask[inside_mask == True] = visibility_mask#inside_mask.shape=(237360,),将inside_mask==true的inside_mask改成visibility_mask,使得inside_mask为ture的点既是第i张照片里的点，也是该视角的可见点。
            visible_points_view[i] = inside_mask
        return visible_points_view, projected_points, resolution
    
    def get_bbox(self, mask, view):
        '''
        获取边界框方法 (get_bbox):

        接受掩膜和视角作为输入。
        根据掩膜和视角获取可见点的边界框。
        返回边界框的有效性和坐标。
        '''
        if(self.visible_points_in_view_in_mask[view][mask].sum()!=0):
            true_values = np.where(self.visible_points_in_view_in_mask[view, mask])
            valid = True
            t, b, l, r = true_values[0].min(), true_values[0].max()+1, true_values[1].min(), true_values[1].max()+1 
        else:
            valid = False
            t, b, l, r = (0,0,0,0)
        return valid, (t, b, l, r)
    
    def get_visible_points_in_view_in_mask(self):
        '''
        获取每个视角中每个掩膜内可见点方法 (get_visible_points_in_view_in_mask):

        根据掩膜和视角计算每个视角中每个掩膜内的可见点。
        将结果存储在 visible_points_in_view_in_mask 中，并更新实例变量。
        返回每个视角中每个掩膜内的可见点、投影点和图像分辨率。
        '''
        masks = self.masks #masks.masks.shape = (237360,148)
        num_view = len(self.indices)#238 #visible_points_view表示在第i张图中，237360个点云投影的可见性, projected_points表示23768个点投影到第i张图片的坐标点,
        visible_points_view, projected_points, resolution = self.get_visible_points_view()# visible_points_view.shape=(238, 237360), projected_points.shape=(238, 237360, 2), resolution=(480,640)
        visible_points_in_view_in_mask = np.zeros((num_view, masks.num_masks), dtype=float)
        image_mask = np.zeros((resolution[0], resolution[1]),dtype=bool)
        # ** visible_points_in_view_in_mask.shape=(238,148,480,640)
        print(f"[INFO] Computing the visible points in each view in each mask.")
        for i in tqdm(range(num_view)):#238
            for j in range(masks.num_masks):#148
                visible_masks_points = (masks.masks[:,j] * visible_points_view[i]) > 0
                proj_points = projected_points[i][visible_masks_points]#proj_points为第i个视角的第j个实例的可见点在图像中的坐标(w,h)
                # print(proj_points.shape) # (0,2)
                if(len(proj_points) != 0):# proj_points.shape=(0,2)：表示在第i视角，第j实例没有可见点
                    image_mask[proj_points[:,1], proj_points[:,0]] = True
                    num_points_in_view_in_mask = image_mask.sum(axis=0).sum(axis=0)
                    visible_points_in_view_in_mask[i][j]=num_points_in_view_in_mask
                image_mask = np.zeros((resolution[0], resolution[1]),dtype=bool)
        self.visible_points_in_view_in_mask = visible_points_in_view_in_mask # visible_points_in_view_in_mask.shape=(238,148,480,640) 每个视角的每个实例的可视点
        self.visible_points_view = visible_points_view # visible_points_view.shape=(238, 237360) 每个视角的可视点
        self.projected_points = projected_points # projected_points.shape=(238, 237360, 2) 每个点云投影到各视角的坐标
        self.resolution = resolution # resolution=(480,640) 分辨率
        return visible_points_in_view_in_mask, visible_points_view, projected_points, resolution
    
    def point_coords_in_view(self,view_index,mask_index):
        masks = self.masks #masks.masks.shape = (237360,148)
        # num_view = len(self.indices)#238 #visible_points_view表示在第i张图中，237360个点云投影的可见性, projected_points表示23768个点投影到第i张图片的坐标点,
        visible_points_view, projected_points, resolution = self.visible_points_view,self.projected_points,self.resolution #get_visible_points_view()# visible_points_view.shape=(238, 237360), projected_points.shape=(238, 237360, 2), resolution=(480,640)
        
        image_mask = np.zeros((resolution[0], resolution[1]),dtype=bool)
        # # ** visible_points_in_view_in_mask.shape=(238,148,480,640)
        # print(f"[INFO] Computing the visible points in each view in each mask.")
        # for i in tqdm(range(num_view)):#238
        #     for j in range(masks.num_masks):#148
        visible_masks_points = (masks.masks[:,mask_index] * visible_points_view[view_index]) > 0
        proj_points = projected_points[view_index][visible_masks_points]#proj_points为第i个视角的第j个实例的可见点在图像中的坐标(w,h)
                # print(proj_points.shape) # (0,2)
        point_coords = np.ones((1,2)).astype('int64')
        if(len(proj_points) != 0):# proj_points.shape=(0,2)：表示在第i视角，第j实例没有可见点
            image_mask[proj_points[:,1], proj_points[:,0]] = True
                    # num_points_in_view_in_mask = image_mask.sum(axis=0).sum(axis=0)
                    # visible_points_in_view_in_mask[i][j]=num_points_in_view_in_mask
                        # Get original mask points coordinates in 2d images
            point_coords = np.transpose(np.where(image_mask == True)) #visible_points_in_view_in_mask.shape=(238,148,480,640)
                
            image_mask = np.zeros((resolution[0], resolution[1]),dtype=bool)

        self.point_coords = point_coords # resolution=(480,640) 分辨率
        return point_coords
    
    def get_visible_points_in_view_in_mask_l(self):
        '''
        获取每个视角中每个掩膜内可见点方法 (get_visible_points_in_view_in_mask):

        根据掩膜和视角计算每个视角中每个掩膜内的可见点。
        将结果存储在 visible_points_in_view_in_mask 中，并更新实例变量。
        返回每个视角中每个掩膜内的可见点、投影点和图像分辨率。
        '''
        masks = self.masks #masks.masks.shape = (237360,148)
        num_view = len(self.indices)#238 #visible_points_view表示在第i张图中，237360个点云投影的可见性, projected_points表示23768个点投影到第i张图片的坐标点,
        visible_points_view, projected_points, resolution = self.get_visible_points_view()# visible_points_view.shape=(238, 237360), projected_points.shape=(238, 237360, 2), resolution=(480,640)
        visible_points_in_view_in_mask = np.zeros((num_view, masks.num_masks, resolution[0], resolution[1]), dtype=bool)
        # ** visible_points_in_view_in_mask.shape=(238,148,480,640)
        print(f"[INFO] Computing the visible points in each view in each mask.")
        for i in tqdm(range(num_view)):#238
            for j in range(masks.num_masks):#148
                visible_masks_points = (masks.masks[:,j] * visible_points_view[i]) > 0
                proj_points = projected_points[i][visible_masks_points]#proj_points为第i个视角的第j个实例的可见点在图像中的坐标(w,h)
                # print(proj_points.shape) # (0,2)
                if(len(proj_points) != 0):# proj_points.shape=(0,2)：表示在第i视角，第j实例没有可见点
                    visible_points_in_view_in_mask[i][j][proj_points[:,1], proj_points[:,0]] = True
        self.visible_points_in_view_in_mask = visible_points_in_view_in_mask # visible_points_in_view_in_mask.shape=(238,148,480,640) 每个视角的每个实例的可视点
        self.visible_points_view = visible_points_view # visible_points_view.shape=(238, 237360) 每个视角的可视点
        self.projected_points = projected_points # projected_points.shape=(238, 237360, 2) 每个点云投影到各视角的坐标
        self.resolution = resolution # resolution=(480,640) 分辨率
        return visible_points_in_view_in_mask, visible_points_view, projected_points, resolution
    
    def get_top_k_indices_per_mask(self, k):
        '''
        获取每个掩膜内的前 k 个可见点的索引方法 (get_top_k_indices_per_mask)：

        根据每个掩膜内的可见点数目，返回前 k 个可见点的索引。
        索引按照可见点数量的降序排列。
        '''
        num_points_in_view_in_mask = self.visible_points_in_view_in_mask #.sum(axis=2).sum(axis=2) #num_points_in_view_in_mask.shape = (238,148)
        topk_indices_per_mask = np.argsort(-num_points_in_view_in_mask, axis=0)[:k,:].T
        print(num_points_in_view_in_mask)
        print(topk_indices_per_mask)
        print(topk_indices_per_mask.shape)
        return topk_indices_per_mask
    
class FeaturesExtractor:
    def __init__(self, 
                 camera, 
                 clip_model, 
                 images, 
                 masks,
                 pointcloud,
                 sam_model_type,
                 sam_checkpoint,
                 vis_threshold,
                 device):
        '''
        接受多个参数，包括相机参数 camera、剪辑模型 clip_model、图像数据 images、掩膜数据 masks、点云数据 pointcloud、SAM模型类型 sam_model_type、SAM检查点 sam_checkpoint、可视化阈值 vis_threshold和设备 device。
        这些参数被存储在实例变量中，以供后续的特征提取过程使用。
        '''
        self.camera = camera
        self.images = images
        self.device = device
        self.point_projector = PointProjector(camera, pointcloud, masks, vis_threshold, images.indices)#PointProjector 类：用于将点云投影到图像平面，并确定哪些点在特定的掩膜内可见。
        self.predictor_sam = initialize_sam_model(device, sam_model_type, sam_checkpoint)
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device)
        
    
    def extract_features(self, topk, multi_level_expansion_ratio, num_levels, num_random_rounds, num_selected_points, save_crops, out_folder, optimize_gpu_usage=False):
        '''
        提取特征功能 (extract_features 方法):

        接受多个参数，包括 topk、multi_level_expansion_ratio、num_levels、num_random_rounds、num_selected_points、save_crops、out_folder 和 optimize_gpu_usage。
        在提取特征的过程中，首先通过 PointProjector 类获取每个掩膜中前 k 个关键点的索引。
        然后，对于每个掩膜，遍历每个视角，进行以下操作：
        运行 SAM 模型以获取最佳掩膜。
        根据给定的层数和扩展比例，生成多级裁剪。
        对于每个裁剪的图像，使用 CLIP 模型提取图像特征。
        将提取的特征进行归一化，并存储在 mask_clip 数组中。
        返回 mask_clip 数组，其中包含了每个掩膜的特征表示。
        '''
        if(save_crops):
            out_folder = os.path.join(out_folder, "crops")
            os.makedirs(out_folder, exist_ok=True)
                            
        topk_indices_per_mask = self.point_projector.get_top_k_indices_per_mask(topk)#(148, 5) 148个实例，选出可视点最多的5个视角，并返回它们的索引
        
        num_masks = self.point_projector.masks.num_masks #148
        mask_clip = np.zeros((num_masks, 768)) #initialize mask clip （148，768）
        
        np_images = self.images.get_as_np_list()
        for mask in tqdm(range(num_masks)): # for each mask 
            images_crops = []
            if(optimize_gpu_usage):
                self.clip_model.to(torch.device('cpu'))
                self.predictor_sam.model.cuda()
            for view_count, view in enumerate(topk_indices_per_mask[mask]): # for each view view=5个图片索引
                if(optimize_gpu_usage):
                    torch.cuda.empty_cache()
                
                # # Get original mask points coordinates in 2d images
                # point_coords = np.transpose(np.where(self.point_projector.visible_points_in_view_in_mask[view][mask] == True)) #visible_points_in_view_in_mask.shape=(238,148,480,640)
                point_coords = self.point_projector.point_coords_in_view(view,mask)
                if (point_coords.shape[0] > 1):
                    self.predictor_sam.set_image(np_images[view])
                    
                    # SAM
                    best_mask = run_sam(image_size=np_images[view],
                                        num_random_rounds=num_random_rounds,
                                        num_selected_points=num_selected_points,
                                        point_coords=point_coords,
                                        predictor_sam=self.predictor_sam,)
                    
                    # MULTI LEVEL CROPS
                    for level in range(num_levels):
                        # get the bbox and corresponding crops
                        x1, y1, x2, y2 = mask2box_multi_level(torch.from_numpy(best_mask), level, multi_level_expansion_ratio)
                        cropped_img = self.images.images[view].crop((x1, y1, x2, y2))
                        
                        if(save_crops):
                            cropped_img.save(os.path.join(out_folder, f"crop{mask}_{view}_{level}.png"))
                            
                        # I compute the CLIP feature using the standard clip model
                        cropped_img_processed = self.clip_preprocess(cropped_img)
                        images_crops.append(cropped_img_processed)
            
            if(optimize_gpu_usage):
                self.predictor_sam.model.cpu()
                self.clip_model.to(torch.device('cuda'))                
            if(len(images_crops) > 0):
                image_input = torch.tensor(np.stack(images_crops))
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input.to(self.device)).float()
                    image_features /= image_features.norm(dim=-1, keepdim=True) #normalize
                
                mask_clip[mask] = image_features.mean(axis=0).cpu().numpy()
                    
        return mask_clip
        
    