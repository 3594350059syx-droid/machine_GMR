import cv2
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp

# ==== 1) MuJoCo 模型加载 ====
model_path = r"E:\machine_GMR\Humanoid\humanoid.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# ==== 2) Pose Landmarker 初始化 ====
pose_model_path = r"E:\machine_GMR\models\pose_landmarker_full.task"
base_options = python.BaseOptions(model_asset_path=pose_model_path)
pose_options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
landmarker = vision.PoseLandmarker.create_from_options(pose_options)

# ==== 3) 被动 viewer ====
viewer = mujoco.viewer.launch_passive(model, data)
viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
scene = viewer.user_scn

# ==== 4) 摄像头 ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

# ==== 5) CSV 保存准备 ====
csv_rows = []
frame_idx = 0

# ==== 6) 坐标转换 & 安全角度 ====
def cam_to_mj(x, y, z, scale=1.0):
    X = (x - 0.5) * scale
    Y = (0.5 - y) * scale
    Z = -z * 0.3
    return np.array([X, Y, Z])

def safe_angle(v1, v2):
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(v1,v2)/(norm1*norm2), -1.0, 1.0)
    return np.arccos(cos_angle)

# ==== 7) 手臂更新函数 ====
def update_arm(data, model, shoulder_pt, elbow_pt, wrist_pt,
               shoulder_name, elbow_name, smoothing=0.2):

    # 上臂向量和前臂向量
    upper = elbow_pt - shoulder_pt
    lower = wrist_pt - elbow_pt

    # 肩膀 hinge 角度 (xz平面)
    def shoulder_hinge(v1, v2):
        v1_xz = np.array([v1[0], v1[2]])
        v2_xz = np.array([v2[0], v2[2]])
        if np.linalg.norm(v1_xz)<1e-6 or np.linalg.norm(v2_xz)<1e-6:
            return 0.0
        return np.arctan2(v2_xz[1], v2_xz[0]) - np.arctan2(v1_xz[1], v1_xz[0])

    shoulder_angle = shoulder_hinge(upper, lower)
    elbow_angle    = safe_angle(upper, lower)

    # 获取 actuator id
    aid_shoulder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, shoulder_name)
    aid_elbow    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, elbow_name)

    # 安全夹角
    def clip_angle(aid, angle):
        if aid == -1:
            return angle
        jid = model.actuator_trnid[aid, 0]
        low, high = model.jnt_range[jid]
        return np.clip(angle, low, high)

    shoulder_angle = clip_angle(aid_shoulder, shoulder_angle)
    elbow_angle    = clip_angle(aid_elbow, elbow_angle)

    # 平滑控制
    if aid_shoulder != -1:
        data.ctrl[aid_shoulder] = data.ctrl[aid_shoulder]*(1-smoothing) + shoulder_angle*smoothing
    if aid_elbow != -1:
        data.ctrl[aid_elbow] = data.ctrl[aid_elbow]*(1-smoothing) + elbow_angle*smoothing

    # 调试输出
    print(f"{shoulder_name}: {np.degrees(shoulder_angle):.1f} deg, "
          f"{elbow_name}: {np.degrees(elbow_angle):.1f} deg")

# ==== 8) 主循环 ====
while viewer.is_running():
    ret, frame = cap.read()
    if not ret:
        print("摄像头帧读取失败")
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        print("未检测到人体关键点")
        continue

    landmarks = result.pose_landmarks[0]
    mj_coords = np.array([cam_to_mj(lm.x, lm.y, lm.z) for lm in landmarks])

    # 更新左右手
    update_arm(data, model, mj_coords[12], mj_coords[14], mj_coords[16],
               "right_shoulder", "right_elbow")
    update_arm(data, model, mj_coords[11], mj_coords[13], mj_coords[15],
               "left_shoulder", "left_elbow")

    # 视频显示关键点
    for lm in landmarks:
        x, y = int(lm.x*frame.shape[1]), int(lm.y*frame.shape[0])
        cv2.circle(frame, (x, y), 4, (0,255,0), -1)

    # 保存 CSV
    row = [frame_idx] + [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
    row.append(np.degrees(safe_angle(mj_coords[12]-mj_coords[14], mj_coords[16]-mj_coords[14]))) # right
    row.append(np.degrees(safe_angle(mj_coords[11]-mj_coords[13], mj_coords[15]-mj_coords[13]))) # left
    csv_rows.append(row)

    cv2.putText(frame, f"Right Elbow: {int(row[-2])} deg", (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, f"Left Elbow: {int(row[-1])} deg", (30,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Pose + Angle", frame)
    frame_idx += 1

    # MuJoCo 更新
    mujoco.mj_step(model, data)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==== 9) 退出 & CSV ====
cap.release()
cv2.destroyAllWindows()
landmarker.close()
viewer.close()

cols = ["frame"] + [f"lm{i}_{ax}" for i in range(33) for ax in ['x','y','z']] + ["right_elbow_angle","left_elbow_angle"]
df = pd.DataFrame(csv_rows, columns=cols)
df.to_csv("pose_mj_output.csv", index=False)
print("CSV 保存完成！pose_mj_output.csv")