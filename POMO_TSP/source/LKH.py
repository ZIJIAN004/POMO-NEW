import numpy as np
import elkai
from scipy.spatial import distance_matrix

def solve_tsp_with_lkh(coords):
    """
    使用 LKH 求解器求解 TSP 问题。
    
    Args:
        coords (np.ndarray): 形状为 (N, 2) 的 numpy 数组，表示 N 个节点的 (x, y) 坐标。
        
    Returns:
        list: 包含构成回环的节点索引列表，例如 [0, 5, 1, ..., 0]
    """
    
    # 1. 计算距离矩阵 (Distance Matrix)
    # LKH 需要知道两两节点之间的距离，coords 只是坐标
    # result shape: (N, N)
    dist_mat = distance_matrix(coords, coords)
    
    # 2. 调用 elkai (LKH 的 Python 封装)
    # solve_float_matrix 专门处理浮点数距离矩阵
    # 这里的 tour 返回的是不包含回环的路径，例如 [0, 5, 2, 1]
    tour = elkai.solve_float_matrix(dist_mat)
    
    # 3. 构成回环
    # 题目要求包含起点构成回环，所以要把第一个点追加到最后
    tour.append(tour[0])
    
    return tour

