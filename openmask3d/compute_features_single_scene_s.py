import hydra
from omegaconf import DictConfig
import numpy as np
from openmask3d.data.load import Camera, InstanceMasks3D, Images, PointCloud, get_number_of_images
from openmask3d.utils import get_free_gpu, create_out_folder
from openmask3d.mask_features_computation.features_extractor import FeaturesExtractor
import torch
import os
import time

from application.pc_processing import read_3d_map
# TIP: add version_base=None to the arguments if you encounter some error
@hydra.main(config_path="configs", config_name="openmask3d_inferences")
def main(ctx: DictConfig):

    device = "cpu" 
    device = get_free_gpu(min_mem=4000) if torch.cuda.is_available() else device
    print(f"Using device: {device}")
    
    out_folder = ctx.output.output_directory
    
    # convert all paths to absolute paths
    os.chdir(hydra.utils.get_original_cwd())
    # ctx.data.masks.masks_path = os.path.abspath(ctx.data.masks.masks_path)#masks.pt
    ctx.data.camera.poses_path = os.path.abspath(ctx.data.camera.poses_path)#pose/
    ctx.data.camera.intrinsic_path = os.path.abspath(ctx.data.camera.intrinsic_path)#intrinsic_color.txt
    ctx.data.depths.depths_path = os.path.abspath(ctx.data.depths.depths_path)#depth/
    ctx.data.images.images_path = os.path.abspath(ctx.data.images.images_path)#color/
    # ctx.data.point_cloud_path = os.path.abspath(ctx.data.point_cloud_path)#scene_example.ply
    ctx.external.sam_checkpoint = os.path.abspath(ctx.external.sam_checkpoint)#sam.pth
    ctx.output.output_directory = os.path.abspath(ctx.output.output_directory)# output/
    map_save_path = '/home/ztl/deeplearn/vlmaps_ithor/vlmaps_dataset/vlmaps_dataset/FloorPlan_Val1_4/vlmap/vlmaps_f.h5df'
    (
    mapped_iter_list,
    grid_feat,
    grid_pos,
    weight,
    occupied_ids,
    grid_rgb,
    mask_array,
        ) = read_3d_map(map_save_path)
    # 1. Load the masks
    # assert os.path.exists(ctx.data.masks.masks_path), f"Path to masks does not exist: {ctx.data.masks.masks_path} - first run compute_masks_single_scene.sh!"
    masks = InstanceMasks3D(mask_array)#masks.masks.shape = (237360,148)
    print(f"[INFO] Masks loaded. {masks.num_masks} masks found.")    
    start = time.time()
    # 2. Load the images
    indices = np.arange(0, get_number_of_images(ctx.data.camera.poses_path), step = ctx.openmask3d.frequency)# shape:(238,) [0,10,20,30...]
    images = Images(images_path=ctx.data.images.images_path, 
                    extension=ctx.data.images.images_ext, 
                    indices=indices)
    print(f"[INFO] Images loaded. {len(images.images)} images found.")#238
    
    # 3. Load the pointcloud
    pointcloud = PointCloud(grid_pos)# pointcloud.points.shape = (237360,3)
    print(f"[INFO] Pointcloud loaded. {pointcloud.num_points} points found.")# 237360
    
    # 4. Load the camera configurations
    camera = Camera(intrinsic_path=ctx.data.camera.intrinsic_path, 
                    intrinsic_resolution=ctx.data.camera.intrinsic_resolution, 
                    poses_path=ctx.data.camera.poses_path, 
                    depths_path=ctx.data.depths.depths_path, 
                    extension_depth=ctx.data.depths.depths_ext, 
                    depth_scale=ctx.data.depths.depth_scale)
    print("[INFO] Camera configurations loaded.")

    # 5. Run extractor
    features_extractor = FeaturesExtractor(camera=camera, 
                                           clip_model=ctx.external.clip_model, 
                                           images=images, 
                                           masks=masks,
                                           pointcloud=pointcloud, 
                                           sam_model_type=ctx.external.sam_model_type,
                                           sam_checkpoint=ctx.external.sam_checkpoint,
                                           vis_threshold=ctx.openmask3d.vis_threshold,
                                           device=device)
    print("[INFO] Computing per-mask CLIP features.")
    features = features_extractor.extract_features(topk=ctx.openmask3d.top_k, # 5
                                                   multi_level_expansion_ratio = ctx.openmask3d.multi_level_expansion_ratio,#0.1
                                                   num_levels=ctx.openmask3d.num_of_levels, #3
                                                   num_random_rounds=ctx.openmask3d.num_random_rounds,#10
                                                   num_selected_points=ctx.openmask3d.num_selected_points,#5
                                                   save_crops=ctx.output.save_crops,#False
                                                   out_folder=out_folder,#output
                                                   optimize_gpu_usage=ctx.gpu.optimize_gpu_usage)
    print("[INFO] Features computed.")
    # 6. Save features
    scene_name = os.path.join(ctx.data.masks.masks_path).split("/")[-1][:-9]
    filename = f"{scene_name}_openmask3d_features.npy"
    output_path = os.path.join(out_folder, filename)
    np.save(output_path, features)
    print(f"[INFO] Masks features saved to {output_path}.")
    end = time.time()
    print("time:",end-start)
    
if __name__ == "__main__":
    main()