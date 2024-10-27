import numpy as np

# 加载 .npy 文件
data = np.load('data/UAV2B_joint_motion.npz')


# 列出文件中的数组名称
print("数组名称:", data.files)

for array_name in data.files:
    print(f"数组 '{array_name}' 的形状:", data[array_name].shape)
#
#
#
# # 查看数据结构
# print("Data Type:", data.dtype)  # 数据类型
# print("Shape:", data.shape)      # 数据的维度
# print("Size:", data.size)        # 数据的总元素数
# #
# # 查看部分内容
# print("First few elements:", data[:5])  # 显示前5个元素（可视数据内容情况调整）