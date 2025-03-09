import logging
import os
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys
)
from pytorch_lightning import Trainer
import open3d as o3d
import numpy as np
import torch
import time
import pdb

def get_parameters(cfg: DictConfig):
    '''
    该函数接受一个配置对象 cfg,加载环境变量和配置文件,并返回模型的配置对象、模型本身和日志记录器。
    首先,加载环境变量和配置文件。
    创建一个实例分割模型 InstanceSegmentation,根据需要加载预训练的骨干网络和整个模型的检查点。
    返回模型的配置对象、模型本身和日志记录器。
    '''
    #logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    #loggers = []

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    #logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, None #loggers


def load_ply(filepath):
    '''
    该函数接受一个点云文件路径,并使用 Open3D 库加载该文件,读取点云数据、颜色和法线。
    返回点云的坐标、颜色和法线。
    '''
    pcd = o3d.io.read_point_cloud(filepath)
    pcd.estimate_normals()
    coords = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals)
    return coords, colors, normals

def process_file(filepath):
    '''
    该函数接受一个点云文件路径,调用 load_ply 函数加载点云数据,并根据需要处理数据。
    返回一个列表,其中包含点云的坐标、特征、标签等信息。
    '''
    coords, colors, normals = load_ply(filepath)
    raw_coordinates = coords.copy()
    raw_colors = (colors*255).astype(np.uint8)
    raw_normals = normals

    features = colors
    if len(features.shape) == 1:
        features = np.hstack((features[None, ...], coords))
    else:
        features = np.hstack((features, coords))

    filename = filepath.split("/")[-1][:-4]
    return [[coords, features, [], filename, raw_colors, raw_normals, raw_coordinates, 0]] # 2: original_labels, 3: none
    # coordinates, features, labels, self.data[idx]['raw_filepath'].split("/")[-2], raw_color, raw_normals, raw_coordinates, idx

@hydra.main(config_path="conf", config_name="config_base_class_agn_masks_single_scene.yaml")
def get_class_agnostic_masks(cfg: DictConfig):
    '''
    该函数使用 Hydra 框架装饰器声明为主函数,接受一个配置对象 cfg。
    根据是否可用 GPU 确定设备。
    调用 get_parameters 函数获取模型配置对象和模型本身。
    使用配置对象中的数据处理器 test_collation 构建输入批处理。
    调用 process_file 函数加载场景点云,并将其转换为模型可接受的批处理格式。
    将模型移至适当的设备,并设置为评估模式。
    使用 torch.no_grad() 上下文管理器,对输入批次进行推理,并获取预测结果。
    打印推理时间。
    '''

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)

    c_fn = hydra.utils.instantiate(cfg.data.test_collation) #(model.config.data.test_collation)

    input_batch = process_file(cfg.general.scene_path)
    batch = c_fn(input_batch)

    model.to(device)
    model.eval()

    start = time.time()
    with torch.no_grad():
        res_dict = model.get_masks_single_scene(batch)
    end = time.time()
    print("Time elapsed: ", end - start)

@hydra.main(config_path="conf", config_name="config_base_class_agn_masks_single_scene.yaml")
def main(cfg: DictConfig):
    get_class_agnostic_masks(cfg)

if __name__ == "__main__":
    main()
