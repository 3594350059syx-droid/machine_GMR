import mujoco
import mujoco.viewer
import math

model = mujoco.MjModel.from_xml_path(r"E:\machine_GMR\Unitree_A1\Humanoid\humanoid.xml")
data = mujoco.MjData(model)

t = 0

# 新版建议直接用 launch_passive，循环中可以更新 ctrl
with mujoco.viewer.launch_passive(model, data) as viewer:
    while True:
        # 左肩 ±0.5 rad
        data.ctrl[1] = 0.5 * math.sin(t)
        # 右肩 ±0.5 rad
        data.ctrl[3] = 0.5 * math.sin(t + math.pi)
        # 左肘 ±0.3 rad
        data.ctrl[2] = 0.3 * math.sin(t)
        # 右肘 ±0.3 rad
        data.ctrl[4] = 0.3 * math.sin(t)

        mujoco.mj_step(model, data)
        t += 0.05