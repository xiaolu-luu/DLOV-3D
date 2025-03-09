from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import gdown

from tqdm import tqdm
import clip
import cv2
import torchvision.transforms as transforms
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
import torch

from vlmaps.utils.clip_utils import get_text_feats_multiple_templates
from vlmaps.utils.visualize_utils import pool_3d_label_to_2d

# from utils.ai2thor_constant import ai2thor_class_list
# from utils.clip_mapping_utils import load_map
# from utils.planning_utils import (
#     find_similar_category_id,
#     get_dynamic_obstacles_map,
#     get_lseg_score,
#     get_segment_islands_pos,
#     mp3dcat,
#     segment_lseg_map,
# )
from vlmaps.map.vlmap_builder import VLMapBuilder
from vlmaps.utils.mapping_utils import load_3d_map
from vlmaps.map.map import Map
from vlmaps.utils.index_utils import find_similar_category_id, get_segment_islands_pos, get_dynamic_obstacles_map_3d
from vlmaps.utils.clip_utils import get_lseg_score

from application.pc_processing import read_3d_map

class VLMap(Map):
    def __init__(self, map_config: DictConfig, data_dir: str = ""):
        super().__init__(map_config, data_dir=data_dir)
        self.scores_mat = None
        self.categories = None

        # TODO: check if needed
        # map_path = os.path.join(map_dir, "grid_lseg_1.npy")
        # self.map = load_map(map_path)
        # self.map_cropped = self.map[self.xmin : self.xmax + 1, self.ymin : self.ymax + 1]
        # self._init_clip()
        # self._customize_obstacle_map(
        #     map_config["potential_obstacle_names"],
        #     map_config["obstacle_names"],
        #     vis=False,
        # )
        # self.obstacles_new_cropped = Map._dilate_map(
        #     self.obstacles_new_cropped == 0,
        #     map_config["dilate_iter"],
        #     map_config["gaussian_sigma"],
        # )
        # self.obstacles_new_cropped = self.obstacles_new_cropped == 0
        # self.load_categories()
        # print("a VLMap is created")
        # pass

    def create_map(self, data_dir: Union[Path, str]) -> None:
        print(f"Creating map for scene at: ", data_dir)
        self._setup_paths(data_dir)#PosixPath('/home/ztl/deeplearning/vlmaps/vlmaps_dataset/vlmaps_dataset/5LpN3gDmAk7_1')
        self.map_builder = VLMapBuilder(
            self.data_dir,
            self.map_config,
            self.pose_path,
            self.rgb_paths,
            self.depth_paths,
            self.base2cam_tf,
            self.base_transform,
        )
        if self.map_config.pose_info.pose_type == "mobile_base":
            self.map_builder.create_mobile_base_map()
        elif self.map_config.pose_info.pose_type == "camera":
            self.map_builder.create_camera_map()

    def load_map(self, data_dir: str) -> bool:
        self._setup_paths(data_dir)
        self.map_save_path = Path(data_dir) / "vlmap" / "vlmaps_f.h5df"
        if not self.map_save_path.exists():
            print("Loading VLMap failed because the file doesn't exist.")
            return False
        (
            self.mapped_iter_list,
            self.grid_feat,
            self.grid_pos,
            self.weight,
            self.occupied_ids,
            self.grid_rgb,
            self.mask_array
        ) = read_3d_map(self.map_save_path)

        return True

    def _init_clip(self, clip_version="ViT-B/32"):
        if hasattr(self, "clip_model"):
            print("clip model is already initialized")
            return
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.clip_version = clip_version
        self.clip_feat_dim = {
            "RN50": 1024,
            "RN101": 512,
            "RN50x4": 640,
            "RN50x16": 768,
            "RN50x64": 1024,
            "ViT-B/32": 512,
            "ViT-B/16": 512,
            "ViT-L/14": 768,
        }[self.clip_version]
        print("Loading CLIP model...")
        self.clip_model, self.preprocess = clip.load(self.clip_version)  # clip.available_models()
        self.clip_model.to(self.device).eval()

    def init_categories(self, categories: List[str]) -> np.ndarray:
        self.categories = categories
        self.scores_mat = get_lseg_score(
            self.clip_model,
            self.categories,
            self.grid_feat,
            self.clip_feat_dim,
            use_multiple_templates=True,
            add_other=True,
        )  # score for name and other
        return self.scores_mat #scores_mat.shape = (198640, 39)

    def index_map_old(self, language_desc: str, with_init_cat: bool = True):
        if with_init_cat and self.scores_mat is not None and self.categories is not None:
            cat_id = find_similar_category_id(language_desc, self.categories)
            scores_mat = self.scores_mat
        else:
            if with_init_cat:
                raise Exception(
                    "Categories are not preloaded. Call init_categories(categories: List[str]) to initialize categories."
                )
            scores_mat = get_lseg_score(
                self.clip_model,
                [language_desc],
                self.grid_feat,
                self.clip_feat_dim,
                use_multiple_templates=True,
                add_other=True,
            )  # score for name and other
            cat_id = 0

        max_ids = np.argmax(scores_mat, axis=1)
        mask = max_ids == cat_id
        return mask
    
    def index_map(self, language_desc: str, with_init_cat: bool = True):
        if with_init_cat and self.scores_mat is not None and self.categories is not None:
            cat_id = find_similar_category_id(language_desc, self.categories)
            scores_mat = self.scores_mat
        else:
            if with_init_cat:
                raise Exception(
                    "Categories are not preloaded. Call init_categories(categories: List[str]) to initialize categories."
                )
            # scores_mat = get_lseg_score(
            #     self.clip_model,
            #     [language_desc],
            #     self.grid_feat,
            #     self.clip_feat_dim,
            #     use_multiple_templates=True,
            #     add_other=True,
            # )  # score for name and other
            # cat_id = 0

            ########
            path_openmask3d_features = "/home/ztl/deeplearn/vlmaps_ithor/output-FloorPlan_Val2_2/m_openmask3d_features.npy"
            query_similarity_computer = QuerySimilarityComputation()
            openmask3d_features = np.load(path_openmask3d_features)
            pred_masks = np.asarray((self.mask_array)).T
            per_mask_query_sim_scores = query_similarity_computer.compute_similarity_scores(openmask3d_features, language_desc)
            score_vlmaps = vlmaps_clip(self.clip_model,language_desc,self.grid_feat)
            
            per_point_score = query_similarity_computer.get_per_point_colors_for_similarity(per_mask_query_sim_scores, pred_masks,score_vlmaps) 
            # 创建一个与x形状相同的mask数组，并初始化为0
            mask = np.zeros(per_point_score.shape, dtype=int)

            # mask[per_point_score > 0.7] = 1

            mask_score = np.zeros(per_point_score.shape, dtype=int)
            top_scores_indices = np.argsort(per_point_score)[-300:][::-1]  # 从大到小排序
            mask_score[per_point_score > 0.7] = 1
            # 在mask中设置这些索引对应的位置为1
            mask[top_scores_indices] = 1
            mask = mask & mask_score
            ########


        return mask, per_point_score


    def customize_obstacle_map(
        self,
        potential_obstacle_names: List[str],
        obstacle_names: List[str],
        vis: bool = False,
    ):
        if self.obstacles_cropped is None and self.obstacles_map is None:
            self.generate_obstacle_map()
        if not hasattr(self, "clip_model"):
            print("init_clip in customize obstacle map")
            self._init_clip()

        self.obstacles_new_cropped = get_dynamic_obstacles_map_3d(
            self.clip_model,
            self.obstacles_cropped,
            self.map_config.potential_obstacle_names,
            self.map_config.obstacle_names,
            self.grid_feat,
            self.grid_pos,
            self.rmin,
            self.cmin,
            self.clip_feat_dim,
            vis=vis,
        )
        self.obstacles_new_cropped = Map._dilate_map(
            self.obstacles_new_cropped == 0,
            self.map_config.dilate_iter,
            self.map_config.gaussian_sigma,
        )# obstacles_new_cropped.shape=(422, 362)
        self.obstacles_new_cropped = self.obstacles_new_cropped == 0

    # def load_categories(self, categories: List[str] = None):
    #     if categories is None:
    #         if self.map_config["categories"] == "mp3d":
    #             categories = mp3dcat.copy()
    #         elif self.map_config["categories"] == "ai2thor":
    #             categories = ai2thor_class_list.copy()

    #     predicts = segment_lseg_map(self.clip_model, categories, self.map_cropped, self.clip_feat_dim)
    #     no_map_mask = self.obstacles_new_cropped > 0  # free space in the map

    #     self.labeled_map_cropped = predicts.reshape((self.xmax - self.xmin + 1, self.ymax - self.ymin + 1))
    #     self.labeled_map_cropped[no_map_mask] = -1
    #     labeled_map = -1 * np.ones((self.map.shape[0], self.map.shape[1]))

    #     labeled_map[self.xmin : self.xmax + 1, self.ymin : self.ymax + 1] = self.labeled_map_cropped

    #     self.categories = categories
    #     self.labeled_map_full = labeled_map

    # def load_region_categories(self, categories: List[str]):
    #     if "other" not in categories:
    #         self.region_categories = ["other"] + categories
    #     predicts = segment_lseg_map(
    #         self.clip_model, self.region_categories, self.map_cropped, self.clip_feat_dim, add_other=False
    #     )
    #     self.labeled_region_map_cropped = predicts.reshape((self.xmax - self.xmin + 1, self.ymax - self.ymin + 1))

    # def get_region_predict_mask(self, name: str) -> np.ndarray:
    #     assert self.region_categories
    #     cat_id = find_similar_category_id(name, self.region_categories)
    #     mask = self.labeled_map_cropped == cat_id
    #     return mask

    # def get_predict_mask(self, name: str) -> np.ndarray:
    #     cat_id = find_similar_category_id(name, self.categories)
    #     return self.labeled_map_cropped == cat_id

    # def get_distribution_map(self, name: str) -> np.ndarray:
    #     assert self.categories
    #     cat_id = find_similar_category_id(name, self.categories)
    #     if self.scores_map is None:
    #         scores_list = get_lseg_score(self.clip_model, self.categories, self.map_cropped, self.clip_feat_dim)
    #         h, w = self.map_cropped.shape[:2]
    #         self.scores_map = scores_list.reshape((h, w, len(self.categories)))
    #     # labeled_map_cropped = self.labeled_map_cropped.copy()
    #     return self.scores_map[:, :, cat_id]

    def get_pos(self, name: str) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], Any]:
        """
        Get the contours, centers, and bbox list of a certain category
        on a full map
        关键函数
        """
        assert self.categories
        # cat_id = find_similar_category_id(name, self.categories)
        # labeled_map_cropped = self.scores_mat.copy()  # (N, C) N: number of voxels, C: number of categories
        # labeled_map_cropped = np.argmax(labeled_map_cropped, axis=1)  # (N,)
        # pc_mask = labeled_map_cropped == cat_id # (N,)
        # self.grid_pos[pc_mask]
        pc_mask,per_point_score = self.index_map(name, with_init_cat=False)#pc_mask = (198640,),得到当前物品的掩码
        mask_2d = pool_3d_label_to_2d(pc_mask, self.grid_pos, self.gs)# mask_2d.shape = (1000, 1000)，将3d掩码投影到2d栅格坐标中        mask_2d = mask_2d[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1]#mask_2d.shape = (422, 362),根据障碍地图截取
        mask_2d = mask_2d[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1]
        # print(f"showing mask for object cat {name}")
        # cv2.imshow(f"mask_{name}", (mask_2d.astype(np.float32) * 255).astype(np.uint8))
        # cv2.waitKey()

        foreground = binary_closing(mask_2d, iterations=3)
        foreground = gaussian_filter(foreground.astype(float), sigma=0.8, truncate=3)
        foreground = foreground > 0.5
        # cv2.imshow(f"mask_{name}_gaussian", (foreground * 255).astype(np.uint8))
        foreground = binary_dilation(foreground)
        # cv2.imshow(f"mask_{name}_processed", (foreground.astype(np.float32) * 255).astype(np.uint8))
        # cv2.waitKey()

        contours, centers, bbox_list, _ = get_segment_islands_pos(foreground, 1)
        # print("centers", centers)

        # whole map position
        for i in range(len(contours)):
            centers[i][0] += self.rmin
            centers[i][1] += self.cmin
            bbox_list[i][0] += self.rmin
            bbox_list[i][1] += self.rmin
            bbox_list[i][2] += self.cmin
            bbox_list[i][3] += self.cmin
            for j in range(len(contours[i])):
                contours[i][j, 0] += self.rmin
                contours[i][j, 1] += self.cmin

        return contours, centers, bbox_list


def vlmaps_clip(clip_model,text_query,seg_data):
    # # seg_data = torch.load(pcd_seg_filepath)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(device)
    # clip_version = "ViT-B/32"
    # clip_feat_dim = {
    #     "RN50": 1024,
    #     "RN101": 512,
    #     "RN50x4": 640,
    #     "RN50x16": 768,
    #     "RN50x64": 1024,
    #     "ViT-B/32": 512,
    #     "ViT-B/16": 512,
    #     "ViT-L/14": 768,
    #     }[clip_version]
    # print("Loading CLIP model...")
    # clip_model, preprocess = clip.load(clip_version,device=device)  # clip.available_models()
    # clip_model.to(device).eval()
    text_input_processed = clip.tokenize(text_query).to(device)#torch.Size([1, 77])
    with torch.no_grad():
        sentence_embedding = clip_model.encode_text(text_input_processed) #torch.Size([1, 768])

    
    sentence_embedding_normalized = (sentence_embedding/sentence_embedding.norm(dim=-1, keepdim=True)).float().cpu()#torch.Size([1, 768])
    sentence_embedding_normalized = sentence_embedding_normalized.squeeze().numpy()
    data_s = seg_data#['feat'] 
    scores3d_list = np.zeros(len(data_s))
    data_norm = np.linalg.norm(data_s,axis=1)
    # 找出范数为0的索引
    zero_norm_indices = data_norm == 0

    # 用1替换范数为0的值，以避免除以0
    data_norm = np.where(zero_norm_indices, 1, data_norm)

    data_norm = data_norm.reshape(-1, 1)
    normalized_emb = (data_s/data_norm)
    scores3d_list = normalized_emb @ sentence_embedding_normalized.T
        # return sentence_embedding_normalized.squeeze().numpy()
    return scores3d_list

def average_score(new_score):
    # 获取x的最小值和最大值
    min_val = np.min(new_score)
    max_val = np.max(new_score)

    # 计算最大值和最小值的差
    range_val = max_val - min_val

    # 进行归一化
    # 避免除以零的情况，如果range_val为0，则直接返回0数组
    if range_val != 0:
        normalized_score = (new_score - min_val) / range_val
    else:
        normalized_score = np.zeros_like(new_score)
    normalized_score =normalized_score.squeeze()
    return normalized_score


class QuerySimilarityComputation():
    def __init__(self,):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.clip_model, _ = clip.load('ViT-L/14@336px', self.device)

    def get_query_embedding(self, text_query):
        text_input_processed = clip.tokenize(text_query).to(self.device)#torch.Size([1, 77])
        with torch.no_grad():
            sentence_embedding = self.clip_model.encode_text(text_input_processed) #torch.Size([1, 768])

        sentence_embedding_normalized =  (sentence_embedding/sentence_embedding.norm(dim=-1, keepdim=True)).float().cpu()#torch.Size([1, 768])
        return sentence_embedding_normalized.squeeze().numpy()
 
    def compute_similarity_scores(self, mask_features, text_query):
        text_emb = self.get_query_embedding(text_query)# shape:(768,) 

        scores = np.zeros(len(mask_features))# shape:(148,) [0.0,0.0,0.0......]
        for mask_idx, mask_emb in enumerate(mask_features):#mask_features.shape = (148, 768) mask_idx=0~147;mask_emb.shape=(768,)
            mask_norm = np.linalg.norm(mask_emb)#0.9649
            if mask_norm < 0.001:
                continue
            normalized_emb = (mask_emb/mask_norm)
            scores[mask_idx] = normalized_emb@text_emb

        return scores # return a scores array,array.shape=(148,)
    
    def get_per_point_colors_for_similarity(self, 
                                            per_mask_scores, 
                                            masks, 
                                            score_vlmaps,
                                            normalize_based_on_current_min_max=False, 
                                            normalize_min_bound=0.16, #only used for visualization if normalize_based_on_current_min_max is False
                                            normalize_max_bound=0.26, #only used for visualization if normalize_based_on_current_min_max is False
                                            background_color=(0.77, 0.77, 0.77)
                                        ):
        # get the per-point heatmap colors for the similarity scores
        # get colors based on the openmask3d per mask scores
        non_zero_points = per_mask_scores!=0 #shape:(148,) [True,True,True,True,...],min=max=True
        openmask3d_per_mask_scores_rescaled = np.zeros_like(per_mask_scores)#shape:(148,)
        pms = per_mask_scores[non_zero_points] #(148,)

        # in order to be able to visualize the score differences better, we can use a normalization scheme
        if normalize_based_on_current_min_max: # if true, normalize the scores based on the min. and max. scores for this scene
            openmask3d_per_mask_scores_rescaled[non_zero_points] = (pms-pms.min()) / (pms.max() - pms.min())
        else: # if false, normalize the scores based on a pre-defined color scheme with min and max clipping bounds, normalize_min_bound and normalize_max_bound.
            new_scores = np.zeros_like(openmask3d_per_mask_scores_rescaled)#shape:(148,) all of 0
            new_indices = np.zeros_like(non_zero_points)#shape:(148,) all of False
            new_indices[non_zero_points] += pms>normalize_min_bound # when pms>normalize_min_bound,new_indices set True
            new_scores[new_indices] = ((pms[pms>normalize_min_bound]-normalize_min_bound)/(normalize_max_bound-normalize_min_bound))#shape:(148,) score normalization,
            openmask3d_per_mask_scores_rescaled = new_scores 

        new_score = np.ones((masks.shape[1], 1))*0 #+ background_color #shape:(237360, 3)
        
        for mask_idx, mask in enumerate(masks[::-1, :]): #mask_idx=0~147 mask.shape = (237360,)
            # get color from matplotlib colormap
            new_score[mask>0.5, :] = openmask3d_per_mask_scores_rescaled[len(masks)-mask_idx-1]
        
        normalized_score = average_score(new_score)
        score_vlmaps_0 = score_vlmaps
        
        score_vlmaps_0 = (score_vlmaps-0.6)/0.1
        score_vlmaps_0[score_vlmaps < 0.6] = 0
        normalized_score_vlmaps = average_score(score_vlmaps_0)#原始的未处理的情况：score_vlmaps

        ave_score = (3*normalized_score+2*normalized_score_vlmaps)/5
        # ave_score[ave_score <0.7 ] = 0.5 

        
        # new_colors = np.ones((masks.shape[1], 3))*0 + background_color #shape:(237360, 3)
        # # print(new_colors.shape)
        # for i in range(len(new_score)): #mask_idx=0~147 mask.shape = (237360,)
        #     # get color from matplotlib colormap
        #     x= ave_score[i]
        #     # print(x)
        #     co = plt.cm.jet(x)[:3]
        #     # print(co,'\n')
        #     new_colors[i, :] = co

        return ave_score