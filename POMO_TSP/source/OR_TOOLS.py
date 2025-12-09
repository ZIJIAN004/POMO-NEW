import numpy as np
from scipy.spatial import distance_matrix
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

class TSPSolver:
    """
    基于 Google OR-Tools 的 TSP 求解器封装。
    包含求解最优路径和计算给定路径长度的功能。
    """

    def __init__(self, time_limit_sec=1, scale_factor=10000):
        """
        初始化求解器。
        
        Args:
            time_limit_sec (int): 求解器的最大运行时间（秒）。
            scale_factor (int): 坐标放大倍数（用于 OR-Tools 整数运算）。
        """
        self.time_limit_sec = time_limit_sec
        self.scale_factor = scale_factor
        self._cached_dists = None  # 用于缓存距离矩阵，避免重复计算

    def _get_distance_matrix(self, coordinates):
        """内部辅助方法：计算或获取缓存的距离矩阵"""
        return distance_matrix(coordinates, coordinates)

    def calculate_path_length(self, coordinates, path):
        """
        [新增功能] 计算给定节点序列的总路径长度。

        Args:
            coordinates (np.ndarray): 节点坐标数组 (N, 2)。
            path (list[int] or np.ndarray): 节点访问顺序索引，例如 [0, 5, 2, 3, 1, 4]。
                                            注意：如果需要闭环（回到起点），path 最后必须包含起点索引。
        
        Returns:
            float: 该路径的总欧几里得距离。
        """
        # 1. 获取距离矩阵
        dists = self._get_distance_matrix(coordinates)
        
        # 2. 累加距离
        total_dist = 0.0
        # 遍历 path，计算 path[i] 到 path[i+1] 的距离
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i+1]
            total_dist += dists[from_node][to_node]
            
        return total_dist

    def solve(self, coordinates):
        """
        求解 TSP 问题。

        Returns:
            list[int]: 最优路径索引序列 (闭环)。
            float: 总距离。
        """
        # --- 1. 输入校验 ---
        if not isinstance(coordinates, np.ndarray) or coordinates.ndim != 2:
            raise ValueError("Input coordinates must be a 2D numpy array.")
        
        num_nodes = len(coordinates)
        if num_nodes < 2:
            return [0, 0], 0.0

        # --- 2. 获取距离矩阵并缩放 ---
        # 使用缓存机制
        dists = self._get_distance_matrix(coordinates)
        # 将浮点距离放大并转为整数 (OR-Tools 要求)
        scaled_dists = (dists * self.scale_factor).astype(int)

        # --- 3. 创建路由模型 (标准 OR-Tools 流程) ---
        manager = pywrapcp.RoutingIndexManager(num_nodes, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return scaled_dists[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # --- 4. 配置参数 ---
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = self.time_limit_sec

        # --- 5. 求解 ---
        solution = routing.SolveWithParameters(search_parameters)

        # --- 6. 提取结果 ---
        if solution:
            route = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                index = solution.Value(routing.NextVar(index))
            
            # 闭环：加上回到起点
            route.append(manager.IndexToNode(index))
            
            # 直接用我们的新方法计算 float 距离，比 OR-Tools 返回的 int 距离更精确
            total_distance = self.calculate_path_length(coordinates, route)
            
            return route, total_distance
        else:
            return None, None

# --- 测试代码 ---
if __name__ == "__main__":
    # 生成 5 个随机点
    coords = np.random.rand(5, 2)
    print("Coordinates:\n", coords)

    solver = TSPSolver(time_limit_sec=1)
    
    # 1. 求解最优路径
    best_route, best_len = solver.solve(coords)
    print(f"\n[OR-Tools] Optimal Route: {best_route}")
    print(f"[OR-Tools] Optimal Length: {best_len:.4f}")

    # 2. 手动测试一个路径 (比如按顺序走 0->1->2->3->4->0)
    manual_route = [0, 1, 2, 3, 4, 0]
    manual_len = solver.calculate_path_length(coords, manual_route)
    print(f"\n[Manual] Route: {manual_route}")
    print(f"[Manual] Calculated Length: {manual_len:.4f}")
    
    # 验证对比
    if best_len <= manual_len:
        print("\n验证通过: OR-Tools 找到的路径确实比（或等于）手动顺序路径短。")
