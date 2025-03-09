import os
import cv2
import numpy as np
from ai2thor.controller import Controller
import time
import random
from pathlib import Path

def keyboard_control_fast():
    k = cv2.waitKey(1)
    if k == ord("a"):
        print('a')
        action = "turn_left"
    elif k == ord("d"):
        action = "turn_right"
    elif k == ord("w"):
        action = "move_forward"
    elif k == ord("s"):
        action = "move_back"
    elif k == ord("q"):
        action = "stop"
    elif k == ord(" "):
        print('record')
        return k, "record"
    elif k == -1:
        return k, None
    else:
        return -1, None
    return k, action

def show_rgb(obs):
    bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    cv2.imshow("rgb", bgr)

def save_image(root_save_dir,image,save_id):
    root_save_dir = Path(root_save_dir)
    save_name = f"{save_id:06}.png" 
    save_dir = root_save_dir / "rgb"
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / save_name
    cv2.imwrite(str(save_path), image[:, :, [2, 1, 0]])

def save_depth(root_save_dir,image,save_id):
    root_save_dir = Path(root_save_dir) 
    save_name = f"{save_id:06}.npy"
    save_dir = root_save_dir / "depth"
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / save_name
    with open(save_path, "wb") as f:
         np.save(f,image)


def main(root_save_dir,scene_name):
    controller = Controller(
        agentMode="LoCoBot",
        visibilityDistance=3.5,
        scene=scene_name, 

        # step sizes
        gridSize=0.01,
        snapToGrid=False,
        rotateStepDegrees=5,

        # image modalities
        renderDepthImage=True,
        renderInstanceSegmentation=True,

        # camera properties
        width=1080,
        height=1080,
        fieldOfView=90
    )
    # controller.step(
    # action="RandomizeLighting",
    # brightness=(0.5, 1.5),
    # randomizeColor=True,
    # hue=(0, 1),
    # saturation=(0.5, 1),
    # synchronized=False
    # )
    positions = controller.step(
        action="GetReachablePositions"
    ).metadata["actionReturn"]
    # init_agent_state = random.choice(positions)
    position_save = []
    position_save_cp = []
    # controller.step(
    #     action="TeleportFull",
    #     position=init_agent_state,
    #     rotation=dict(x=0,y=0,z=0),
    #     horizon=0,
    #     # standing =True
    #     )
    # position_save.append({"position":controller.last_event.metadata['agent']['position'],"rotation":controller.last_event.metadata['agent']['rotation']})
    last_action = None
    root_save_dir = str(root_save_dir + scene_name)
    print(root_save_dir)
    while True:
        obs = controller.last_event.frame
        show_rgb(obs)
        k, action = keyboard_control_fast()
        if k != -1:
            if action == "stop":
                break
            if action == "record":
                init_agent_state = {"position":controller.last_event.metadata['agent']['position'],"rotation":controller.last_event.metadata['agent']['rotation']}
                continue
            last_action = action
            release_count = 0
        else:
            if last_action is None:
                time.sleep(0.01)
                continue
            else:
                release_count += 1
                if release_count > 1:
                    print("stop after release")
                    last_action = None
                    release_count = 0
                    continue
                action = last_action
        if action=='turn_right':
            controller.step(action="RotateRight",degrees=5)
        elif action == 'turn_left':
            controller.step(action="RotateLeft",degrees=5)
        elif action == "move_back":
            controller.step(action="MoveBack",moveMagnitude=0.1)
        else :
            controller.step(action="MoveAhead",moveMagnitude=0.1)
        # actions_list.append(action)
        agent_state = {"position":controller.last_event.metadata['agent']['position'],"rotation":controller.last_event.metadata['agent']['rotation']}
        position_save.append(agent_state)
        print(len(position_save))
    
    position_save_cp = position_save[::3] 
    print(position_save_cp[0:2])
    print(len(position_save_cp))
    # np.save("position.npy",position_save_cp)
    for i in range (len(position_save_cp)):
        if i == 500:
            break  
        event = controller.step(
        action="TeleportFull",
        position=position_save_cp[i]["position"],
        rotation=position_save_cp[i]["rotation"],
        horizon=0,
        # standing =True
        ) 
        position=position_save_cp[i]["position"]
        
        with open(root_save_dir+'/poses.txt', 'a') as file:
            # 随机选择一个position字典
            position['qy'] = position_save_cp[i]["rotation"]['y']
            # 将字典的值转换为列表，并用逗号分隔
            values = [str(value) for value in position.values()]
            # 将值列表转换为字符串，用逗号分隔，并写入文件
            file.write(' '.join(values) + '\n')
        image_rgb = event.frame 
        save_image(root_save_dir,image_rgb,i)
        image_depth = event.depth_frame
        save_depth(root_save_dir,image_depth,i)   


if __name__ == "__main__":
    scene_name = "FloorPlan_Val2_4"
    root_save_dir = '/home/ztl/deeplearn/vlmaps_ithor/vlmaps_dataset/vlmaps_dataset/'
    main(root_save_dir,scene_name)