# DLOV-3D
Dual-Level Open-Vocabulary 3D Scene Representation for  Zero-Shot Object Navigation 

we propose a novel **Dual**-**L**evel **O**pen-**V**ocabulary **3D**  (DLOV-3D) scene representation framework to improve robot navigation performance. Our framework integrates both pixel-level and image-level features into spatial scene representations, facilitating a more comprehensive understanding of the scene. By incorporating an adaptive revalidation mechanism, DLOV-3D achieves precise instance-aware navigation based on free-form queries that describe object properties such as color, shape, and object references.
# Approach

![](.\image\framework.png)
# Visual Demo
- Evaluation in ai2thor

<p align="center">
    <img src=".\image\1.gif" alt="Framework">
</p>

- Evaluation in Habitat

<p align="center">
    <img src=".\image\2.gif" alt="Framework">
</p>

# Quick Start

## Dependencies Installation
Download this repository locally
```bash
git clone https://github.com/xiaolu-luu/DLOV-3D.git
```

The repository needs to use [ai2thor](https://github.com/allenai/ai2thor) as the simulator.
```
pip install ai2thor
pip install -r requirements.txt
``` 
Once you've installed AI2-THOR, you can verify that everything is working correctly by running the following minimal example:
```bash
from ai2thor.controller import Controller
controller = Controller(scene="FloorPlan10")
event = controller.step(action="RotateRight")
metadata = event.metadata
print(event, event.metadata.keys())
```
## Generate Dataset
To construct DLOV-3D in the ai2thor simulation environment, we manually collected 10 RGB-D video sequences from 10 scenes in the ai2thor simulator. We provide scripts to capture the RGB-D sequences. Please follow the steps below to generate the dataset.
```bash
python data_collection.py
```
the structure of the ai2thor_scene_dir looks like this :
```
dataset_scene_dir
          |-FloorPlan_Val2_2
          |   |-depth
          |   |-rgb
          |   |-poses.txt
          |   |-...
          |-FloorPlan_Val2_3
          |   |-depth
          |   |-rgb
          |   |-poses.txt
          |   |-...
          |-FloorPlan_Val2_4
          |   |-depth
          |   |-rgb
          |   |-poses.txt
          |   |-...
          ...
```
## Create a DLOV-3D with the Generated Dataset
* Change the value for `defaults/data_paths` in `config/map_creation_cfg.yaml` to `default`.
* Change the `ai2thor_scene_dir` and `dlov_data_dir` in `config/data_paths/default.yaml` according to the steps in the **Generate Dataset** section above.
* Run the following command to build the VLMap. The code builds a 3D map where each voxel contains the LSeg embedding.
```bash
cd DLOV-3D
sh create.sh
```

## DLOV-3D Visualization

* Change the value for `defaults/data_paths` in `config/map_indexing_cfg.yaml` to `default`.
* Change the `ai2thor_scene_dir` and `dlov_data_dir` in `config/data_paths/default.yaml` according to the steps in the **Generate Dataset** section above.
* Run the following command to visualize the DLOV-3D you built
```bash
cd openmask3d/visualization
python viz_sim_score_result.py
```
## Test Navigation

### Setup OpenAI
In order to test object goal navigation and spatial goal navigation tasks with our method, you need to setup an OpenAI API account with the following steps:
1. [Sign up an OpenAI account](https://openai.com/blog/openai-api), login your account, and bind your account with at least one payment method.
2. [Get you OpenAI API keys](https://platform.openai.com/account/api-keys), copy it.
3. Open your `~/.bashrc` file, paste a new line `export OPENAI_KEY=<your copied key>`, and save the file.


### Run Object Goal Navigation

1. Run object goal navigation. The code will load tasks specified in `<scene_folder>/object_navigation_tasks.json`. The results will be saved in `<scene_folder>/dlov_obj_nav_results/`. 
```bash
cd application/evaluation
python evaluate_object_goal_navigation.py
```
### TODO: Compute the Final Metrics




