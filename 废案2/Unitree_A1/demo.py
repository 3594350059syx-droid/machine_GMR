import mujoco
import mujoco.viewer
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path(r"E:\machine_GMR\mujoco_menagerie\unitree_a1\scene.xml")
data = mujoco.MjData(model)

# 重置数据
mujoco.mj_resetData(model, data)

# 设置 Base 初始位置和朝向
data.qpos[0:3] = np.array([0, 0, 0.43])  # Base xyz
data.qpos[3:7] = np.array([1, 0, 0, 0])  # Base 四元数正立

# PD 控制参数
kp = 50
kd = 2

# 按名称获取所有腿关节
leg_joints = [
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'
]

# 获取每个关节在 data.qpos 和 data.qvel 中的索引
joint_ids = [model.joint(jname).id for jname in leg_joints]

# 记录腿部初始姿态
qpos_leg0 = data.qpos.copy()

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 相机跟随机器人躯干
    try:
        trunk_id = model.body('trunk').id
        viewer.cam.trackbodyid = trunk_id
    except Exception:
        print("无法跟踪 trunk，相机使用默认视角")

    while viewer.is_running():
        # Base 姿态稳定：锁定四元数
        data.qpos[3:7] = np.array([1, 0, 0, 0])
        data.qvel[0:6] = 0  # Base 线速度+角速度清零

        # PD 控制腿关节
        for jid in joint_ids:
            pos_err = qpos_leg0[jid] - data.qpos[jid]
            vel_err = -data.qvel[jid]
            data.ctrl[jid - 7] = kp*pos_err + kd*vel_err  # ctrl数组从0开始

        mujoco.mj_step(model, data)
        viewer.sync()