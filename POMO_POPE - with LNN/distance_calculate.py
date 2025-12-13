import numpy as np

def calculate_tour_distance(coords, tour):
    """
    计算给定路径序列的总欧式距离。
    
    Args:
        coords (np.ndarray): 节点的坐标数组。形状为 (N, 2)。
                             例如: [[0.1, 0.5], [0.3, 0.8], ...]
        tour (list or np.ndarray): 访问节点的顺序索引序列。
                                   例如: [0, 5, 1, 2, 0] (注意：如果要算闭环，首尾必须相同)
        
    Returns:
        float: 路径的总距离。
    """
    # 1. 按照 tour 的顺序，从 coords 里把坐标取出来
    # 这一步叫 "Advanced Indexing" (花式索引)
    # ordered_coords 的形状将会是 (len(tour), 2)
    ordered_coords = coords[tour]
    
    # 2. 计算相邻点之间的向量差 (Delta)
    # ordered_coords[1:] 是所有“终点”
    # ordered_coords[:-1] 是所有“起点”
    # 结果形状: (len(tour)-1, 2)
    deltas = ordered_coords[1:] - ordered_coords[:-1]
    
    # 3. 计算每个向量的长度 (即每一段的距离)
    # np.hypot(dx, dy) 等价于 sqrt(dx^2 + dy^2)，但更安全、更快
    segment_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    
    # 4. 求和得到总距离
    total_distance = np.sum(segment_lengths)
    
    return float(total_distance)

