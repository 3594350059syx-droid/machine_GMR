import mujoco
import mujoco.viewer
import numpy as np
import time

##机器人本地路径(model保存机器人的模型信息，data是状态)
model = mujoco.MjModel.from_xml_path("E:\machine_GMR\single_joint\pendulum.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:

    t = 0

    while viewer.is_running():
        ##控制电机(>0右转，<0左转)
        data.ctrl[0] = 1

        ##相当于神经，将信号传递给机器人的肢体，并根据物理规律更新状态
        ##仿真时间，模型参数不变，步长不变（基于物理引擎，类似于积分）
        mujoco.mj_step(model, data)
        ##显示
        viewer.sync()
        ##控制信号的时间变化（数字时间，相当于方程未知数）
        t += 0.02
        ##真实时间延迟，便于显示观察
        time.sleep(0.01)