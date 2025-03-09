import numpy as np
from typing import List, Dict, Tuple, Set, Union
from pathlib import Path
import matplotlib.pyplot as plt
def load_depth_npy(depth_filepath: Union[Path, str]):
    with open(depth_filepath, "rb") as f:
        depth = np.load(f)
    return depth

depth_path ='/home/ztl/deeplearn/vlmaps-master/vlmaps_dataset/vlmaps_dataset/test_data/depth/000002.npy'
depth = load_depth_npy(depth_path)
print(depth.shape)

depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

# # 使用 matplotlib 显示深度图
# plt.imshow(depth_normalized, cmap='viridis')  # 可以选择不同的颜色映射，如 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
# # plt.colorbar()  # 显示颜色条
# # plt.title('Depth Visualization')
# plt.savefig('depth_image.png')
# plt.show()


import numpy as np
import cv2  # 导入 OpenCV 库

# 假设 load_depth_npy 函数和 depth 数组已经定义和加载

# 归一化深度图
depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

# 转换为 8 位数据类型
depth_uint8 = (depth_normalized * 255).astype(np.uint8)

# 应用颜色映射
depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PARULA)# COLORMAP_JET

# 显示深度图
cv2.imshow('Depth Visualization', depth_colormap)

# 保存图片到文件系统
# cv2.imwrite('depth_image49_p.png', depth_colormap)

# 等待用户按键操作，0 表示无限等待，直到用户按下任意键
cv2.waitKey(0)

# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()