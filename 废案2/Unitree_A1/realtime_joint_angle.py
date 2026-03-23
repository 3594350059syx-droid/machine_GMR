import cv2
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp  

# ==== 1) MuJoCo 模型加载 ====
model_path = r"E:\machine_GMR\Unitree_A1\Humanoid\humanoid.xml"  # 替换为你的宇树机器人/其他机器人模型
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, data)
viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
# ==== 2.5) 创建33个关键点marker ====
max_markers = 33
scene = viewer.user_scn
# ==== 2) Pose Landmarker 初始化 ====
pose_model_path = "models/pose_landmarker_full.task"
base_options = python.BaseOptions(model_asset_path=pose_model_path)
pose_options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
landmarker = vision.PoseLandmarker.create_from_options(pose_options)

# ==== 3) 摄像头 ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

# ==== 4) CSV 保存准备 ====
csv_rows = []
frame_idx = 0

# ==== 5) IK & 坐标转换函数 ====
def cam_to_mj(x, y, z, scale=1.5):

    X = (x - 0.5) * scale
    Y = (z) * scale
    Z = (1 - y) * scale

    return np.array([X, Y, Z])

def elbow_angle_3d(shoulder, elbow, wrist):

    upper = shoulder - elbow
    lower = wrist - elbow

    upper = upper / np.linalg.norm(upper)
    lower = lower / np.linalg.norm(lower)

    cos_angle = np.dot(upper, lower)
    cos_angle = np.clip(cos_angle, -1, 1)

    angle = np.arccos(cos_angle)

    return angle

def draw_landmarks_mujoco(coords):
    """
    在MuJoCo中绘制关键点
    """
    scene.ngeom = 0

    for p in coords:

        if scene.ngeom >= max_markers:
            break

        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.03, 0, 0],
            pos=p,
            mat=np.eye(3).flatten(),
            rgba=[1, 0, 0, 1]
        )

        scene.ngeom += 1

# ==== 6) 主循环 ====
while viewer.is_running():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        # ==== 6a) 坐标转换到 MuJoCo ====
        mj_coords = []
        for lm in landmarks:
            mj_coords.append(cam_to_mj(lm.x, lm.y, lm.z))
        mj_coords = np.array(mj_coords)
        draw_landmarks_mujoco(mj_coords)
        # ==== 6b) 简单 IK 映射右臂 ====
        shoulder = mj_coords[12]
        elbow = mj_coords[14]
        wrist = mj_coords[16]
        right_elbow_angle = elbow_angle_3d(shoulder, elbow, wrist)

        # ==== 6c) 更新 MuJoCo 关节角 ====
        # 这里只更新右肘关节示例，实际根据模型调整关节名称
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_elbow")

        if jid != -1:
            qpos_id = model.jnt_qposadr[jid]
            data.qpos[qpos_id] = right_elbow_angle

        # ==== 6d) 可视化关键点到摄像头画面 ====
        for lm in landmarks:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 4, (0,255,0), -1)
        angle_deg = np.degrees(right_elbow_angle)
        cv2.putText(frame, f"Right Elbow: {int(angle_deg)} deg",
            (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # ==== 6e) 保存 CSV ====
        row = [frame_idx]
        for lm in landmarks:
            row += [lm.x, lm.y, lm.z]
        row.append(right_elbow_angle)
        csv_rows.append(row)

    # ==== 6f) 显示摄像头画面 ====
    cv2.imshow("Pose + Angle", frame)
    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # ==== 6g) 更新仿真器 ====
    mujoco.mj_step(model, data)

# ==== 7) 退出与保存 CSV ====
cap.release()
cv2.destroyAllWindows()
landmarker.close()
viewer.close()

cols = ["frame"] + [f"lm{i}_{ax}" for i in range(33) for ax in ['x','y','z']] + ["right_elbow_angle"]
df = pd.DataFrame(csv_rows, columns=cols)
df.to_csv("pose_mj_output.csv", index=False)
print("CSV 保存完成！pose_mj_output.csv")