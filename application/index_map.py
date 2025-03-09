from pathlib import Path
import hydra
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.visualize_utils import (
    pool_3d_label_to_2d,
    pool_3d_rgb_to_2d,
    visualize_rgb_map_3d,
    visualize_masked_map_2d,
    visualize_heatmap_2d,
    visualize_heatmap_3d,
    visualize_masked_map_3d,
    get_heatmap_from_mask_2d,
    get_heatmap_from_mask_3d,
)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig) -> None:#/home/ztl/deeplearning/vlmaps/vlmaps_dataset
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"#/Users/huang/hcg/projects/vln/data/vlmaps_dataset'
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    vlmap = VLMap(config.map_config, data_dir=data_dirs[config.scene_id])
    vlmap.load_map(data_dirs[config.scene_id])
    visualize_rgb_map_3d(vlmap.grid_pos, vlmap.grid_rgb)
    #cat = input("What is your interested category in this scene?")
    cat = 'door' #"sofa" 

    vlmap._init_clip()
    print("considering categories: ")
    print(mp3dcat[1:-1])
    if config.init_categories:
        vlmap.init_categories(mp3dcat[1:-1])
        mask = vlmap.index_map(cat, with_init_cat=False)#这里生成的mask是标签对应的掩码
    else:
        mask = vlmap.index_map(cat, with_init_cat=False)# mask.shape = (198640,)

    if config.index_2d:
        mask_2d = pool_3d_label_to_2d(mask, vlmap.grid_pos, config.params.gs) #mask_3d.shape = (1000, 1000)
        rgb_2d = pool_3d_rgb_to_2d(vlmap.grid_rgb, vlmap.grid_pos, config.params.gs) #rgb_2d.shape = (1000, 1000, 3), vlmap.grid_rgb.shape = (198640, 3), vlmap.grid_pos.shape = (198640, 3)
        visualize_masked_map_2d(rgb_2d, mask_2d)
        heatmap = get_heatmap_from_mask_2d(mask_2d, cell_size=config.params.cs, decay_rate=config.decay_rate)
        visualize_heatmap_2d(rgb_2d, heatmap)
    else:
        visualize_masked_map_3d(vlmap.grid_pos, mask, vlmap.grid_rgb)
        heatmap = get_heatmap_from_mask_3d(
            vlmap.grid_pos, mask, cell_size=config.params.cs, decay_rate=config.decay_rate
        )
        visualize_heatmap_3d(vlmap.grid_pos, heatmap, vlmap.grid_rgb)


if __name__ == "__main__":
    main()
