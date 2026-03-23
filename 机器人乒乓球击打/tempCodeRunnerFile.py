import mujoco, mujoco.viewer, time, numpy as np

# --- 1. 速度调节器 ---
# 1.0 是正常速度，0.1 是超级慢动作（适合观察击球瞬间）
time_scale = 0.5 

def check_hit(model, data, ball_geom_name='ball_geom', paddle_geom_name='paddle_geom'):
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ball_geom_name)
    paddle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, paddle_geom_name)
    for i in range(data.ncon):
        contact = data.contact[i]
        if (contact.geom1 == ball_id and contact.geom2 == paddle_id) or \
           (contact.geom1 == paddle_id and contact.geom2 == ball_id):
            return True
    return False

m = mujoco.MjModel.from_xml_path('h1_simple.xml')
d = mujoco.MjData(m)

# 初始发球设置
def reset_ball(data):
    data.joint('ball_joint').qpos[:3] = [2.5, 0, 1.5]
    # 你可以在这里改球的初始速度
    data.joint('ball_joint').qvel[:3] = [-2.0, 0, 3.0] 

reset_ball(d)
hit_count = 0
already_hit_this_round = False 

with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        step_start = time.time()
        
        # 物理步进
        mujoco.mj_step(m, d)

        # --- 核心：确保以下所有代码都在 while 循环的缩进内 ---
        
        # A. 碰撞检测
        if not already_hit_this_round:
            if check_hit(m, d):
                hit_count += 1
                already_hit_this_round = True
                print(f"💥 漂亮！击中球了！总次数: {hit_count}")

        # B. 获取位置数据
        b_p = d.geom('ball_geom').xpos
        s_p = d.body('r_shoulder').xpos
        rel_x = b_p[0] - s_p[0]
        rel_y = b_p[1] - s_p[1]
        rel_z = b_p[2] - s_p[2]

        # C. 空间区域判定与挥拍
        in_strike_zone = (0.2 < rel_x < 0.9) and (-0.6 < rel_y < 0.6) and (-0.7 < rel_z < 0.3)

        if in_strike_zone:
            # 1. 瞄准高度 (Joint 0)
            target_h = rel_z * 1.5 - 0.2
            d.ctrl[0] = np.clip(target_h, -1.2, 1.2)
            
            # 2. 蓄力与爆发逻辑
            if rel_x > 0.45: # 稍微调小一点点，让球更近时再爆发
                # --- 蓄力阶段 ---
                d.ctrl[1] = 0.8    # 肘部向外打开
                d.ctrl[2] = 0.0    # 缩回手臂
            else:
                # --- 击球爆发 (瞬间出拳) ---
                d.ctrl[1] = -4.5   # 肘部快速合拢（横向力量）
                d.ctrl[2] = 0.3    # 手臂瞬间向前弹出（向前推力）
                
                if not already_hit_this_round:
                    print(f"🚀 砰！向前抽击！X剩余距离: {rel_x:.2f}")
        # D. 同步查看器
        v.sync()

        # E. 时间流速控制
        # 通过修改 sleep 时间，让画面变慢
        expected_step_time = m.opt.timestep / time_scale
        elapsed = time.time() - step_start
        if expected_step_time > elapsed:
            time.sleep(expected_step_time - elapsed)

        # F. 重置逻辑
        if b_p[2] < 0.05 or rel_x < -0.3:
            already_hit_this_round = False
            reset_ball(d)
            mujoco.mj_forward(m, d)