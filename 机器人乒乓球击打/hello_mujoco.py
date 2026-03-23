import mujoco, mujoco.viewer, time, numpy as np
import os

# 必须使用与训练时完全一致的类结构
class RobustBrain:
    def __init__(self, sizes=[6, 64, 64, 32, 6]):
        self.weights = [np.zeros((sizes[i], sizes[i+1])) for i in range(len(sizes)-1)]
        self.biases = [np.zeros((1, sizes[i+1])) for i in range(len(sizes)-1)]
        self.running_mean = np.zeros(sizes[0])
        self.running_var = np.ones(sizes[0])
        self.last_action = np.zeros(sizes[-1])

    def swish(self, x):
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -8, 8))))

    def forward(self, state):
        # 注意：这里的统计量会随着测试继续微调，或者你可以保存训练时的统计量
        self.running_mean = 0.999 * self.running_mean + 0.001 * state
        self.running_var = 0.999 * self.running_var + 0.001 * (state - self.running_mean)**2
        norm_state = (state - self.running_mean) / (np.sqrt(self.running_var) + 1e-7)
        
        x = norm_state
        for i in range(len(self.weights) - 1):
            x = self.swish(np.dot(x, self.weights[i]) + self.biases[i])
        
        action = np.tanh(np.dot(x, self.weights[-1]) + self.biases[-1])
        self.last_action = 0.8 * self.last_action + 0.2 * action.flatten()
        return self.last_action

# 1. 加载模型和数据
m = mujoco.MjModel.from_xml_path('h1_simple.xml')
d = mujoco.MjData(m)
brain = RobustBrain()

# 2. 载入权重
weights_path = "e:/machine_GMR/best_h1_weights.npz"
data = np.load(weights_path)
for i in range(len(brain.weights)):
    brain.weights[i] = data[f'w{i}']
    brain.biases[i] = data[f'b{i}']

# 3. 运行演示
with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        # 重置球的位置（可以设置得更难一些）
        mujoco.mj_resetData(m, d)
        d.joint('ball_joint').qpos[:3] = [2.2, np.random.uniform(-0.5, 0.5), 1.3]
        d.joint('ball_joint').qvel[:3] = [-4.5, 0, 1.2]
        
        for _ in range(1000): # 每一球演示 1000 步
            start_time = time.time()
            
            # 获取状态并决策
            obs = np.concatenate([d.geom('ball_geom').xpos, d.joint('ball_joint').qvel[:3]])
            action = brain.forward(obs)
            
            # 控制机器人
            d.ctrl[1] = action[0] * 0.8
            d.ctrl[0] = action[1] * 1.5
            d.ctrl[2:6] = action[2:6] * 1.8
            
            mujoco.mj_step(m, d)
            v.sync()
            
            # 控制演示速度与真实时间同步
            elapsed = time.time() - start_time
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)
            
            # 如果球落地或飞走，提前重置
            if d.geom('ball_geom').xpos[2] < 0.1 or d.geom('ball_geom').xpos[0] < -0.5:
                break