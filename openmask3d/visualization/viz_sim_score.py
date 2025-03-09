import numpy as np
import open3d as o3d
import torch
import clip
import pdb
import matplotlib.pyplot as plt
from constants import *

from application.pc_processing import read_3d_map
from vlmaps.utils.clip_utils import get_text_feats
from vlmaps.utils.matterport3d_categories import mp3dcat,mp3dcat_s
def vlmaps_clip(text_query,seg_data):
    # seg_data = torch.load(pcd_seg_filepath)
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

    text_feats = get_text_feats(text_query, clip_model, clip_feat_dim)
    data_s = seg_data
    scores3d_list = data_s @ text_feats.T
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

def open_clip(text_query,seg_data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    clip_version = "ViT-L/14@336px"
    clip_feat_dim = {
        "RN50": 1024,
        "RN101": 512,
        "RN50x4": 640,
        "RN50x16": 768,
        "RN50x64": 1024,
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14@336px": 768,
        }[clip_version]
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(clip_version,device=device)  # clip.available_models()
    clip_model.to(device).eval()

    text_feats = get_text_feats(text_query, clip_model, clip_feat_dim)
    data_s = seg_data
    scores3d_list = data_s @ text_feats.T
    return scores3d_list

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
        ave_score[ave_score <0.8 ] = 0.5 

        
        new_colors = np.ones((masks.shape[1], 3))*0 + background_color #shape:(237360, 3)
        # print(new_colors.shape)
        for i in range(len(new_score)): #mask_idx=0~147 mask.shape = (237360,)
            # get color from matplotlib colormap
            x= ave_score[i]
            # print(x)
            co = plt.cm.jet(x)[:3]
            # print(co,'\n')
            new_colors[i, :] = co

        return new_colors
        



def main():
    # --------------------------------
    # Set the paths
    # --------------------------------
    # path_scene_pcd = "/home/ztl/deeplearning/vlmaps-demo/data/5LpN3gDmAk7_1/map/scene_pcd_w_sim_colors_gg.ply" #"resources/scene_example/scene_example.ply"
    # path_pred_masks = "/home/ztl/deeplearning/vlmaps-demo/data/5LpN3gDmAk7_1/map/mask_gg.pt" #"output/2024-05-13-22-42-53-experiment/scene_example_masks.pt"
    path_openmask3d_features = "output-jh4fc5c5qoQ_3/m_openmask3d_features.npy" #"output/scene_example_openmask3d_features.npy"
    # pcd_seg_filepath = '/home/ztl/deeplearning/vlmaps-demo/data/5LpN3gDmAk7_1/map/instance_feat_gg.pth'
    map_save_path = '/home/ztl/deeplearning/vlmaps/vlmaps_dataset/vlmaps_dataset/jh4fc5c5qoQ_3/vlmap/vlmaps_f.h5df'
    (
    mapped_iter_list,
    grid_feat,
    grid_pos,
    weight,
    occupied_ids,
    grid_rgb,
    mask_array,
        ) = read_3d_map(map_save_path)
    # --------------------------------
    # Load data
    # --------------------------------
    # load the scene pcd
    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Assign coordinates to the PointCloud object
    pcd.points = o3d.utility.Vector3dVector(grid_pos)
    scene_pcd = pcd#o3d.io.read_point_cloud(pcd)
    
    # load the predicted masks
    pred_masks = np.asarray((mask_array)).T # (num_instances, num_points) shape:(148, 237360) array 0:no 1:yes

    # load the openmask3d features
    openmask3d_features = np.load(path_openmask3d_features) # (num_instances, 768) shape:(148, 768)


    open_feature = np.zeros((pred_masks.shape[1],openmask3d_features.shape[1]))

    for i, row in enumerate(pred_masks):
        # 检查x中的1的位置
        indices = np.where(row == 1)[0]
        # 对于每个索引，将y中的相应数据复制到z中
        for idx in indices:
            open_feature[idx] = openmask3d_features[i]
    # # initialize the query similarity computer
    # query_similarity_computer = QuerySimilarityComputation()
    

    # --------------------------------
    # Set the query text
    # --------------------------------
    query_text = 'Black soft chair in the corner'#'faucet Toilet  Mirror plant staircase white chair '#'Washing machine '#"refrigerator\\window" # change the query text here

    # query_text_vlmap = ['Black soft chair in the corner','A white chair with a scarf','The wooden chair in front of the desk','other']#'White plastic chair'
    query_text_vlmap = mp3dcat_s#['chair','sofa','table','wall',]
    # --------------------------------
    # 'The mirror in the bathroom'
    # 'many black chairs in the living room'
    # 'White plastic chairs under the stairs'White plastic chair under the stairs ‘A white chair with a cloth’
    #'Black wooden chair'
    #'The wooden chair in front of the desk'
    #'A white chair with a scarf'
    #''
    
    # --------------------------------Kitchen sink


    # --------------------------------
    # Get the similarity scores
    # --------------------------------
    # get the per mask similarity scores, i.e. the cosine similarity between the query embedding and each openmask3d mask-feature for each object instance
    # per_mask_query_sim_scores = query_similarity_computer.compute_similarity_scores(openmask3d_features, query_text)#shape:(148,) 148个得分
    score_vlmaps = vlmaps_clip(query_text_vlmap,grid_feat)
    print(score_vlmaps.shape)
    score_open = open_clip(query_text_vlmap,open_feature)
    print(score_open.shape)
    ave_score= (3*score_open+2*score_vlmaps)/5

    predicts3d = np.argmax(ave_score, axis=1)
    print(predicts3d.shape)

    # Get the unique labels in seg_data
    unique_labels = np.unique(predicts3d)#shape=(74,)  min=-1,max=73

    # Generate random colors for each label
    color_map = {label: np.random.rand(3) for label in unique_labels}# len=74

    # Create an array for colors using the color_map
    colors = np.array([color_map[label] for label in predicts3d]) #(232453, 3)



    # # Create a PointCloud object
    # pcd = o3d.geometry.PointCloud()
    # # Assign coordinates to the PointCloud object
    # pcd.points = o3d.utility.Vector3dVector(coordinates)
    # # Assign colors to the PointCloud object
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    # --------------------------------
    # Visualize the similarity scores
    # --------------------------------
    # get the per-point heatmap colors for the similarity scores
    # per_point_similarity_colors = query_similarity_computer.get_per_point_colors_for_similarity(per_mask_query_sim_scores, pred_masks,score_vlmaps) # note: for normalizing the similarity heatmap colors for better clarity, you can check the arguments for the function get_per_point_colors_for_similarity
        # per_point_similarity_colors.shape = (237360, 3)   per_mask_query_sim_scores：shape:(148,)   pred_masks：shape:(148, 237360) array 0:no 1:yes
    
    
    # visualize the scene with the similarity heatmap
    scene_pcd_w_sim_colors = o3d.geometry.PointCloud()
    scene_pcd_w_sim_colors.points = scene_pcd.points
    scene_pcd_w_sim_colors.colors = o3d.utility.Vector3dVector(colors)
    scene_pcd_w_sim_colors.estimate_normals()


    o3d.visualization.draw_geometries([scene_pcd_w_sim_colors],
                                    zoom=0.8600000000000001,
                                    front=[ 0.31489093167413501, -0.10210311276109869, 0.94361997409650555 ],
                                    lookat=[ 512.63165574727566, 600.56741510107713, 13.427799616499735 ],
                                    up=[ -0.86346697954644269, 0.38193477206000914, 0.32947018852153132 ],)
                                    
    # alternatively, you can save the scene_pcd_w_sim_colors as a .ply file
    # o3d.io.write_point_cloud("data/scene_pcd_w_sim_colors_{}.ply".format('_'.join(query_text.split(' '))), scene_pcd_w_sim_colors)

if __name__ == "__main__":
    main()


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
			"front" : [ 0.90785142899306592, -0.2239770510781425, 0.35445742123079788 ],
			"lookat" : [ 402.67641708365318, 500.77127308773987, 48.002339497072541 ],
			"up" : [ -0.32950390174045147, 0.14167776794500461, 0.93346375870086162 ],
			"zoom" : 0.73999999999999999
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
'''

