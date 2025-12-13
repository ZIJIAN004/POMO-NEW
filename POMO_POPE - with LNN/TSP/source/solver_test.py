import dashscope
from OR_TOOLS import * # 如果您本地没有这个文件，请保留下面的 TSPSolver 类定义
import numpy as np
import json
import re
from dashscope.api_entities.dashscope_response import Role

# 【修复1】API Key 必须是字符串
dashscope.api_key = "sk-da4ab8523f6a4f8696cd8f5ee241d9cf"

class NodeEnvironment:
    def __init__(self, num_nodes=5):
        self.num_nodes = num_nodes
        # 随机生成节点坐标 (x, y)
        self.nodes = np.random.rand(num_nodes, 2) * 100 
    
    def calculate_distance(self, path_indices):
        # 1. 基础鲁棒性检查
        if not path_indices or not isinstance(path_indices, list):
            return float('inf')
        
        # 复制一份路径，以免修改原列表
        check_path = path_indices[:]
        
        # ==========================================
        # 【新增加的逻辑】检测非法重复（回环）
        # ==========================================
        
        # 步骤A: 如果路径首尾相同（模型听话地闭环了），先去掉最后一个节点进行检查
        # 例如 [0, 1, 2, 0] -> 变为 [0, 1, 2] 检查唯一性
        if len(check_path) > 1 and check_path[0] == check_path[-1]:
            check_path.pop()
            
        # 步骤B: 检查核心路径是否有重复节点
        # 如果 len([0, 1, 2, 1]) != len({0, 1, 2})，说明中间有重复，路径无效
        if len(check_path) != len(set(check_path)):
            # print(f"检测到非法重复路径: {path_indices}") # 调试用
            return float('inf')

        # 步骤C: 检查是否访问了所有节点
        if len(set(check_path)) != self.num_nodes:
            # print(f"节点缺失或过多: {path_indices}") # 调试用
            return float('inf')

        # ==========================================
        # 计算距离 (兼容闭环和非闭环的写法)
        # ==========================================
        total_dist = 0
        try:
            # 计算点与点之间的距离
            for i in range(len(path_indices) - 1):
                p1 = self.nodes[path_indices[i]]
                p2 = self.nodes[path_indices[i+1]]
                total_dist += np.linalg.norm(p1 - p2)
            
            # 如果模型给的是 [0, 1, 2] (没闭环)，需要手动加上 2->0 的距离
            # 如果模型给的是 [0, 1, 2, 0] (已闭环)，最后一段 0->0 距离为0，不影响结果，所以这里可以统一加上
            if path_indices[0] != path_indices[-1]:
                 total_dist += np.linalg.norm(self.nodes[path_indices[-1]] - self.nodes[path_indices[0]])
                 
        except IndexError:
            return float('inf') # 防止模型输出不存在的ID（比如一共5个点，它输出了 index 10）
            
        return total_dist

    def get_prompt_text(self):
        """将节点数据转化为 Prompt 文本"""
        node_str = "\n".join([f"节点ID {i}: 坐标 ({n[0]:.1f}, {n[1]:.1f})" for i, n in enumerate(self.nodes)])
        return node_str

solver = TSPSolver()

def solve_with_qwen(env):
    prompt_content = f"""
你是一个运筹学优化专家。请解决下面的旅行商问题 (TSP)。
目标：从任意节点出发，访问所有节点一次并回到初始节点，使得总路径最短。

【节点数据】
{env.get_prompt_text()}

【要求】
1. 不需要解释过程。
2. 必须且只能输出一个JSON列表，包含访问节点的ID顺序。
3. 格式示例：[0, 2, 1, 3, 4, 0], 注意末尾需要包含初始节点，形成闭环。
"""
    messages = [{'role': Role.USER, 'content': prompt_content}]
    
    try:
        response = dashscope.Generation.call(
            model='qwen-plus',
            messages=messages,
            result_format='message',
        )
        if response.status_code == 200:
            return response.output.choices[0]['message']['content']
        else:
            print(f"API Error: {response.code}")
            return ""
    except Exception as e:
        print(f"API Exception: {e}")
        return ""

def main():
    time = 10
    # 建议先测试小规模
    node_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] 
    result = {}
    
    for n_nodes in node_list:
        print(f"\n========== 测试节点数: {n_nodes} ==========")
        reward_gap = []
        success_count = 0
        
        for t in range(time):
            print(f"  [Round {t+1}/{time}]")
            env = NodeEnvironment(num_nodes=n_nodes)
            
            # 传统求解
            traditional_path,_ = solver.solve(env.nodes)
            print(traditional_path)
            traditional_reward = env.calculate_distance(traditional_path)
            # print(f"    传统解: {traditional_reward:.2f}")
            
            # 千问求解
            raw_response = solve_with_qwen(env)
            
            # 解析
            qwen_reward = float('inf')
            try:
                match = re.search(r'\[.*?\]', raw_response, re.DOTALL)
                if match:
                    qwen_path = json.loads(match.group())
                    qwen_reward = env.calculate_distance(qwen_path)
                else:
                    print("    解析失败: 无JSON")
            except Exception as e:
                print(f"    解析异常: {e}")

            # 只有当 qwen_reward 不是无穷大时，才算成功
            if qwen_reward != float('inf'):
                diff = (qwen_reward - traditional_reward) / traditional_reward
                reward_gap.append(diff)
                success_count += 1
                print(f"    ✅ 成功! Gap: {diff:.2%}")
                print(qwen_path)
            else:
                # 可能是因为没闭环，也可能是因为中间有非法重复
                print("    ❌ 失败: 路径无效 (包含非法重复或节点缺失)")

        result[n_nodes] = reward_gap
        
        if len(reward_gap) > 0:
            avg_gap = np.mean(reward_gap)
            print(f"--> {n_nodes}节点 成功率: {success_count}/{time}, 平均Gap: {avg_gap:.4f}")
        else:
            print(f"--> {n_nodes}节点 全部失败")

    print("\n========== 最终结果汇总 ==========")
    for n_nodes, gaps in result.items():
        if len(gaps) > 0:
            print(f'{n_nodes}节点的平均 gap 为 {np.mean(gaps):.4f}')
        else:
            print(f'{n_nodes}节点: 无有效数据')

if __name__ == "__main__":
    main()