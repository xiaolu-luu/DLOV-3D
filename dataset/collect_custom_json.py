# from ai2thor.controller import Controller

# controller = Controller(
#     agentMode="default",
#     visibilityDistance=3.5,
#     scene="FloorPlan212", #

#     # step sizes
#     gridSize=0.01,
#     snapToGrid=False,
#     rotateStepDegrees=90,

#     # image modalities
#     renderDepthImage=True,
#     renderInstanceSegmentation=True,

#     # camera properties
#     width=1080,
#     height=1080,
#     fieldOfView=90
# )

# positions = controller.step(
#     action="GetReachablePositions"
# ).metadata["actionReturn"]

# for obj in controller.last_event.metadata["objects"]:
#     print(obj["name"],obj['position'])
'''
ArmChair_60160c64 {'x': -0.27000540494918823, 'y': 0.0002913177013397217, 'z': 1.8700162172317505}
ArmChair_a682f615 {'x': 2.6559951305389404, 'y': 0.0002919137477874756, 'z': 1.8580114841461182}
Boots_611d5bc0 {'x': 4.0045881271362305, 'y': 0.0011577904224395752, 'z': 1.7010477781295776}
Box_38ff2bc7 {'x': -0.47411420941352844, 'y': 1.0353775024414062, 'z': -0.7133176922798157}
CoffeeTable_efe4814d {'x': 1.5910000801086426, 'y': -0.0006985962390899658, 'z': 0.4460012912750244}
CreditCard_b0849dff {'x': 1.4068232774734497, 'y': 0.4715268611907959, 'z': 0.6533510684967041}
Drawer_6a7e335d {'x': 3.8828020095825195, 'y': 0.7749223709106445, 'z': 0.8649265766143799}
Floor_b5efb1ac {'x': 0.0, 'y': 0.0, 'z': 0.0}
FloorLamp_f3968981 {'x': 3.6089677810668945, 'y': 0.00041306018829345703, 'z': 2.16355299949646}
GarbageCan_6f055690 {'x': 3.8339881896972656, 'y': -0.029392749071121216, 'z': -0.4951942265033722}
HousePlant_f4acf145 {'x': 0.38799983263015747, 'y': 0.801343560218811, 'z': -0.7257569432258606}
KeyChain_4960cc52 {'x': 1.501550316810608, 'y': 0.47215741872787476, 'z': 0.5305465459823608}
Laptop_8f86c26a {'x': 1.805005669593811, 'y': 0.47116827964782715, 'z': 0.49900513887405396}
LightSwitch_9447fa35 {'x': -1.3999998569488525, 'y': 1.2899999618530273, 'z': 1.8350000381469727}
Newspaper_941608bd {'x': 2.145167827606201, 'y': 0.4076341390609741, 'z': -0.7197234034538269}
Painting_dfda4f25 {'x': 4.070577144622803, 'y': 1.9509999752044678, 'z': 0.8489999771118164}
Pen_4d33b339 {'x': 3.930530071258545, 'y': 0.8737337589263916, 'z': 1.03994882106781}
Pencil_c25d6da1 {'x': 3.8936095237731934, 'y': 0.8728175759315491, 'z': 1.1799671649932861}
Pillow_6e6a8b47 {'x': 0.6510940194129944, 'y': 0.39362555742263794, 'z': 1.7089107036590576}
RemoteControl_b88fb0ad {'x': 1.8829971551895142, 'y': 0.32785099744796753, 'z': 1.7309972047805786}
Shelf_bd3befef {'x': -0.2899671196937561, 'y': 0.5886980295181274, 'z': -0.7322751879692078}
Shelf_4296f6ab {'x': 1.9054977893829346, 'y': 0.2009461224079132, 'z': -0.7299872040748596}
Shelf_fb6a6b01 {'x': -0.2899581789970398, 'y': 0.20094406604766846, 'z': -0.7322219014167786}
Shelf_a3017c9b {'x': 1.9054994583129883, 'y': 0.5887000560760498, 'z': -0.7300084829330444}
SideTable_bf78fee4 {'x': 3.9469988346099854, 'y': 0.0001315474510192871, 'z': 0.8649976253509521}
Sofa_d438d385 {'x': 1.1930046081542969, 'y': 0.008215636014938354, 'z': 1.8650307655334473}
Statue_7fdc4456 {'x': -0.08691728115081787, 'y': 0.02765485644340515, 'z': -0.6962450742721558}
Statue_f062dfbe {'x': -0.5352799296379089, 'y': 0.40357130765914917, 'z': -0.6937753558158875}
Television_c4df9239 {'x': 1.8979761600494385, 'y': 1.280923843383789, 'z': -0.8382456302642822}
TissueBox_62c595b3 {'x': 3.917987108230591, 'y': 0.8691356182098389, 'z': 0.6810032725334167}
TVStand_28fc8d20 {'x': -0.29023873805999756, 'y': -0.0004183948040008545, 'z': -0.7722402215003967}
TVStand_edafd4ab {'x': 1.9049968719482422, 'y': -0.00041303038597106934, 'z': -0.770020067691803}
WateringCan_37eb4196 {'x': 1.6212167739868164, 'y': 0.02098391205072403, 'z': -0.6973605751991272}
Window_cb736e8a {'x': 1.5700000524520874, 'y': 2.072000026702881, 'z': 2.484999895095825}
Window_bde8e018 {'x': 0.023000000044703484, 'y': 2.072000026702881, 'z': 2.484999895095825}
'''
import numpy as np
import math
def degrees_to_radians(degrees):
    return math.radians(degrees)

def rotation_matrix_y(angle):
    rad = degrees_to_radians(angle)
    return np.array([
        [math.cos(rad), 0, math.sin(rad)],
        [0, 1, 0],
        [-math.sin(rad), 0, math.cos(rad)],
    ])
def agent_state2tf(agent_state):
    tf = np.eye(4)
    tf[0, 3] = agent_state['position']['x']
    tf[1, 3] = agent_state['position']['y']
    tf[2, 3] = agent_state['position']['z']
    quat = agent_state['rotation']['y']
    # r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
    # rot = r.as_matrix()
    # tf[:3, :3] = rot
    tf[:3, :3] = rotation_matrix_y(quat)
    return tf

def cvt_pose_vec2tf(pos_quat_vec: np.ndarray) -> np.ndarray:
    """
    pos_quat_vec: (px, py, pz, qx, qy, qz, qw)
    """
    pose_tf = np.eye(4)
    pose_tf[:3, 3] = pos_quat_vec[:3].flatten()
    m = (pos_quat_vec[3:].flatten())
    pose_tf[:3, :3] = rotation_matrix_y(m[0])
    # rot = R.from_euler('y',m[0])
    # rot = R.from_quat(pos_quat_vec[3:].flatten())
    # pose_tf[:3, :3] = rot.as_matrix()
    return pose_tf

def base_rot_mat2theta(rot_mat: np.ndarray) -> float:
    """Convert base rotation matrix to rotation angle (rad) assuming x is forward, y is left, z is up

    Args:
        rot_mat (np.ndarray): (3,3) rotation matrix

    Returns:
        float: rotation angle
    """
    theta = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    return theta
def base_pos2grid_id_3d(gs, cs, x_base, y_base, z_base):
    '''关键函数,将全局点云转换成矩阵坐标中'''
    row = int(gs / 2 - int(x_base / cs))
    col = int(gs / 2 - int(y_base / cs))
    h = int(z_base / cs)
    return [row, col, h]
def from_habitat_tf(tf_hab: np.ndarray,base_transform,inv_init_base_tf):
    tf = inv_init_base_tf @ base_transform @ tf_hab @ np.linalg.inv(base_transform)
    theta = base_rot_mat2theta(tf[:3, :3])
    theta_deg = np.rad2deg(theta)
    x, y, z = tf[:3, 3]
    gs=1000
    cs=0.05
    row, col, height = base_pos2grid_id_3d(gs, cs, x, y, z)
    full_map_pose = [row, col, theta_deg]
    return full_map_pose
def tar_dict2array(position):
    pos = [] 
    x = position['x']
    y = position['y']
    z = position['z']
    pos = [x,y,z,0]
    return np.array(pos)

# 2.7199997901916504 0.9009991884231567 -0.3999999761581421
'''
FloorLamp_f3968981 {'x': 3.6089677810668945, 'y': 0.00041306018829345703, 'z': 2.16355299949646}  [495, 584]
Box_38ff2bc7 {'x': -0.47411420941352844, 'y': 1.0353775024414062, 'z': -0.7133176922798157} [474, 487]
Pillow_6e6a8b47 {'x': 0.6510940194129944, 'y': 0.39362555742263794, 'z': 1.7089107036590576} [509, 526]
Television_c4df9239 {'x': 1.8979761600494385, 'y': 1.280923843383789, 'z': -0.8382456302642822} [453, 529]
'''
if __name__ == "__main__":
    pose_path = '/home/ztl/deeplearn/vlmaps_ithor/vlmaps_dataset/vlmaps_dataset/FloorPlan212/poses.txt'
    tar_pose_dict =  {'x': 1.8979761600494385, 'y': 1.280923843383789, 'z': -0.8382456302642822}
    base_transform = np.array([[0 ,0,-1,0],
                               [-1,0, 0,0],
                               [0 ,1, 0,0],
                               [0 ,0, 0,1]])
    base_poses = np.loadtxt(pose_path)
    init_base_tf = (
            base_transform @ cvt_pose_vec2tf(base_poses[0]) @ np.linalg.inv(base_transform)
            )
    inv_init_base_tf = np.linalg.inv(init_base_tf)
    full_map_pose = from_habitat_tf(cvt_pose_vec2tf(base_poses[0]),base_transform,inv_init_base_tf)
    print(full_map_pose)

    tar_pose = tar_dict2array(tar_pose_dict)
    tar_hab_tf = cvt_pose_vec2tf(tar_pose)
    tar_full_map_pose = from_habitat_tf(tar_hab_tf,base_transform,inv_init_base_tf)
    print(tar_full_map_pose[:2])

