import numpy as np
from scipy.optimize import minimize

# 模拟奖励函数
def reward_function(period, task_sizes):
    total_size = np.sum(task_sizes)
    noise = np.random.normal(0, 0.1)
    return -0.01 * total_size / period + noise

# 拉格朗日目标函数
def lagrangian(period, task_sizes, min_period, max_period, lambda1, lambda2):
    reward = reward_function(period, task_sizes)
    penalty_min = lambda1 * (period - min_period) if period < min_period else 0
    penalty_max = lambda2 * (max_period - period) if period > max_period else 0
    return -reward + penalty_min + penalty_max

# 参数设置
arms = [1, 5, 10, 20]  # 更新周期候选集
min_period = 1          # 最小周期
max_period = 20         # 最大周期
lambda1 = 1.0           # 拉格朗日乘子
lambda2 = 1.0           # 拉格朗日乘子
context_dim = 10        # 任务类型数量
task_sizes = np.random.randint(50, 300, size=context_dim)

# 优化过程
def optimize_lagrangian(task_sizes):
    result = minimize(
        fun=lambda p: lagrangian(p, task_sizes, min_period, max_period, lambda1, lambda2),
        x0=np.mean(arms),  # 初始值
        bounds=[(min_period, max_period)]  # 边界约束
    )
    return result.x[0], -result.fun  # 返回最优周期和最大化的奖励值

# 模拟
rounds = 100
for t in range(rounds):
    optimal_period, max_reward = optimize_lagrangian(task_sizes)
    print(f"Round {t+1}: Optimal Period = {optimal_period:.4f}, Max Reward = {max_reward:.4f}")
