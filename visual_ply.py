import open3d as o3d
print("Load a ply point cloud, print it, and render it")
path_scene ='/home/ztl/deeplearn/semantic-abstraction/visualization/3/scene_rgb.ply'
path_scene_pcd = "/home/ztl/deeplearn/semantic-abstraction/visualization/3/the door on the stairs.ply"
pcd_scene = o3d.io.read_point_cloud(path_scene)
pcd = o3d.io.read_point_cloud(path_scene_pcd)
o3d.visualization.draw_geometries([pcd_scene,pcd],)
                                #   zoom=0.3412,
                                #   front=[0.4257, -0.2125, -0.8795],
                                #   lookat=[2.6172, 2.0475, 1.532],
                                #   up=[-0.0694, -0.9768, 0.2024])
