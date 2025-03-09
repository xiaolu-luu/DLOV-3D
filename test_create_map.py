# import numpy as np

# p_local = np.array([0.26293635, 0.73852639, 1.13135958])

# radial_dist_sq = np.sum(np.square(p_local))
# sigma_sq = 0.6
# alpha = np.exp(-radial_dist_sq / (2 * sigma_sq))

# x = p_local.shape[0]
# print(x)



# import numpy as np

# # 原始数组
# x = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
#              [0, 1, 0, 1, 0, 1, 0, 1, 0, 0]])
# y = np.array([[3.5,4.5],[4.5,5.5]])

# z = np.zeros(len(x.T), dtype=int)
# print(z)

# # for i in range(len(x)):
# #     zx[i]=1)=y[i]


import numpy as np

# # 给定的数组
# x = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
#               [0, 1, 0, 1, 0, 1, 0, 1, 0, 0]])
# y = np.array([[3.5, 4.5], [4.5, 5.5]])


# # 初始化z数组，形状为(10, 2)，初始值为0
# z = np.zeros((10, 2))

# # 遍历x的转置，这样每行代表一个索引
# for i, row in enumerate(x):
#     # 检查x中的1的位置
#     indices = np.where(row == 1)[0]
#     # 对于每个索引，将y中的相应数据复制到z中
#     for idx in indices:
#         z[idx] = y[i]

# print(z)



# x = np.array([[3.5, 4.5], [4.5, 5.5]])
# y = np.array([[3.5, 4.5], [4.5, 5.5]])
# ave_score = (3*x+2*y)/5

# print(ave_score)






# import numpy as np

# # 假设depth是一个形状为(720, 1080)的NumPy数组
# # 假设我们想要将边缘100像素宽的数据设置为零
# edge_width = 2
# depth = np.array([[1,1,1,1,1,1,1],
#                  [1,1,1,1,1,1,1],
#                  [1,1,1,1,1,1,1],
#                  [1,1,1,1,1,1,1],
#                  [1,1,1,1,1,1,1],
#                  [1,1,1,1,1,1,1],
#                  [1,1,1,1,1,1,1],])
# print(depth.shape)
# # 将左边和右边各100像素宽的数据设置为零
# # depth[0:edge_width, :] = 0
# # depth[-1:-edge_width:-1, :] = 0
# # 创建一个掩码，用于选择边缘像素
# mask = np.zeros(depth.shape, dtype=bool)

# # 设置左边和右边各100像素宽的数据为零
# mask[:, :edge_width] = True  # 左边
# mask[:, -edge_width:] = True  # 右边

# # 应用掩码到depth数组
# depth[mask] = 0

# # 打印结果查看
# print(depth)






# pose_1 = np.array([[1 ,0, 0,1],
#                    [0, 1, 0,2],
#                    [0 ,0, 1,3],
#                    [0 ,0, 0,1]])

# base_transform = np.array([[0 ,0,-1,0],
#                            [-1,0, 0,0],
#                            [0 ,1, 0,0],
#                            [0 ,0, 0,1]])
# inv = np.linalg.inv(base_transform)
# print(inv)

# pose_2 = base_transform @ pose_1 @ inv

# base2cam_tf = np.array([[1 ,0 , 0 ,0],
#                         [0 ,-1, 0 ,1.5],
#                         [0 ,0 ,-1 ,0],
#                         [0 ,0 , 0 ,1]])
# print(pose_2)
# pose_cm=np.array([[1],[2],[3],[1]])
# pose_3 = base2cam_tf@pose_cm

# print(pose_3)



# import numpy as np

# # 假设score是一个长度为100的数组，这里用随机数来模拟
# np.random.seed(0)  # 设置随机种子以保证结果可复现
# score = np.random.randint(0, 101, size=20)  # 生成0到100之间的随机分数

# # 初始化mask数组，形状与score相同，类型为整数
# mask = np.zeros(score.shape, dtype=int)

# # 找出得分前20且分数大于90的索引
# top_scores_indices = (score > 80).nonzero()[0][:5]

# # 在mask中设置这些索引对应的位置为1
# mask[top_scores_indices] = 1

# # 打印结果查看
# print("score:", score)
# print("mask:", mask)

# score: [44 47 64 67 67  9 83 21 36 87 70 88 88 12 58 65 39 87 46 88]
# mask:  [ 0  0  0  0  0  0 1  0  0  1   0 1   1  0  0  0 0  1  0  0]
# mask:  [ 0  0  0  0  0  0 0  0  0  1   0 1   1  0  0  0 0  1  0  1]
# mask:  [ 0  0  0  1  1  0 1  0  0  1   1 1   1  0  0  1 0  1  0  1]
# mask:  [ 0  0  0  0  0  0 1  0  0  1   0 1   1  0  0  0 0  1  0  1]
import numpy as np

# 假设score是一个长度为100的数组，这里用随机数来模拟
np.random.seed(0)  # 设置随机种子以保证结果可复现
score = np.random.randint(0, 101, size=20)  # 生成0到100之间的随机分数

# 初始化mask数组，形状与score相同，类型为整数
mask = np.zeros(score.shape, dtype=int)
mask_score = np.zeros(score.shape, dtype=int)
# 使用argsort找到score中最大的20个元素的索引
top_scores_indices = np.argsort(score)[-10:][::-1]  # 从大到小排序
# mask_score = mask
mask_score[score > 80] = 1
# 在mask中设置这些索引对应的位置为1
mask[top_scores_indices] = 1
mask = mask & mask_score
print(mask_score)
print(mask)
# 打印结果查看
print("score:", score)
print("mask:", mask)