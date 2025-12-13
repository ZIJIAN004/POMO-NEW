import numpy as np
from concorde.tsp import TSPSolver

def solve_tsp_with_concorde(coords):
    """
    使用 Concorde 精确求解器求解 TSP 问题。
    
    Args:
        coords (np.ndarray): 形状为 (N, 2) 的 numpy 数组，浮点数坐标。
        
    Returns:
        list: 包含构成回环的节点索引列表，例如 [0, 5, 1, ..., 0]
    """
    
    # 1. 数据预处理
    # Concorde 的 C 接口对于浮点数的精度处理比较敏感
    # 我们把 xs 和 ys 分开传进去
    xs = coords[:, 0]
    ys = coords[:, 1]
    
    # 2. 初始化求解器
    # norm="EUC_2D" 表示使用二维欧氏距离 (Euclidean 2D)
    # Concorde 会自动把浮点坐标乘以一个系数转为整数进行计算（为了保证精度）
    solver = TSPSolver.from_data(xs, ys, norm="EUC_2D")
    
    # 3. 求解
    # verbose=True 可以看到求解过程日志，设为 False 则静默
    solution = solver.solve(verbose=False)
    
    # solution.tour 是最佳路径的节点索引列表 (不含回环)
    # 例如: [0, 2, 1, 4, 3]
    tour = solution.tour.tolist()
    
    # 4. 构成回环
    # 手动把起点追加到末尾
    tour.append(tour[0])
    
    return tour
