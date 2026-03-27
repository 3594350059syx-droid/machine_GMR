import cv2
import mediapipe as mp
from collections import deque
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# 21个点的名字
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",       # 0-4  拇指
    "INDEX_MCP","INDEX_PIP","INDEX_DIP","INDEX_TIP",       # 5-8  食指
    "MIDDLE_MCP","MIDDLE_PIP","MIDDLE_DIP","MIDDLE_TIP",   # 9-12 中指
    "RING_MCP","RING_PIP","RING_DIP","RING_TIP",           # 13-16 无名指
    "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP",       # 17-20 小指
]

# 每根手指的 [指根MCP索引, 指尖TIP索引]
FINGER_PAIRS = {
    "食指": (5,  8),
    "中指": (9,  12),
    "无名": (13, 16),
    "小指": (17, 20),
}

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:

    frame_count = 0
    history = deque(maxlen=5)
    wrist_history = deque(maxlen=5)
    thumb=0.02
    THRESHOLD = 0.05
    prev_x = 0.0  
    prev_y = 0.0  
    swinging = False
    fingers = [False, False, False, False, False]
    gesture = "unknown"  
    dire=0
    dire_label=""
    GESTURES = {
    "scissors": [None, True, True, False, False],
    "ok":       [None, False, True, True, True],
            }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        finger_count = 0
        stable_count = 0
        if results.multi_hand_landmarks:
            lm_list = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label

            # ── 观察点A：左右手标签 ──
            print(f"[A] 检测到：{handedness}")

            # ① 提取坐标
            lm = []
            for i in lm_list.landmark:
                lm.append((i.x, i.y, i.z))

            wrist_history.append(lm[0])
            smooth_x = sum(p[0] for p in wrist_history) / len(wrist_history)
            smooth_y = sum(p[1] for p in wrist_history) / len(wrist_history)
            cv2.circle(frame, (int(smooth_x * 640), int(smooth_y * 480)), 10, (0, 0, 255), -1)
            speed = abs(smooth_x - prev_x) + abs(smooth_y - prev_y)
            dire = (smooth_x-prev_x) + (smooth_y-prev_y)
            #是否摆动的检测逻辑
            swinging = speed > 0.05
            prev_x = smooth_x
            prev_y = smooth_y
            # ── 观察点B：手腕和食指尖的坐标 ──
            print(f"[B] 手腕    lm[0]  = {lm[0]}")
            print(f"[B] 食指根  lm[5]  = {lm[5]}")
            print(f"[B] 食指尖  lm[8]  = {lm[8]}")

            # ② 循环判断四根手指
            finger_count = 0
            
            for idx, (name, (mcp_i, tip_i)) in enumerate(FINGER_PAIRS.items()):
                is_up = (lm[mcp_i][1] - lm[tip_i][1])>THRESHOLD
                fingers[idx+1]=is_up
                # ── 观察点C：每根手指的判断过程 ──
                print(f"[C] {name}: mcp_y={lm[mcp_i][1]:.3f}  "
                    f"tip_y={lm[tip_i][1]:.3f}  "
                    f"伸出={is_up}")

                if is_up:
                    finger_count += 1
            
            # ③ 拇指,以摄像头为出发点
            if handedness == "Left":
                thumb_up = (lm[3][0] - lm[4][0])>thumb
                print(f"[D详细] handedness={handedness}")
                print(f"[D详细] lm[4].x={lm[4][0]:.3f}  lm[3].x={lm[3][0]:.3f}")
                print(f"[D详细] 公式结果: {lm[3][0] - lm[4][0]:.3f}  THRESHOLD={thumb}")
                print(f"[D详细] thumb_up={thumb_up}")
            else:
                thumb_up = (lm[3][0] - lm[4][0])>thumb
                print(f"[D详细] handedness={handedness}")
                print(f"[D详细] lm[4].x={lm[4][0]:.3f}  lm[3].x={lm[3][0]:.3f}")
                print(f"[D详细] 公式结果: {lm[3][0] - lm[4][0]:.3f}  THRESHOLD={thumb}")
                print(f"[D详细] thumb_up={thumb_up}")
            #判断摆动方向
            if handedness =="Left" and swinging:
                dire_label="Left" if dire>0 else "Right"
            elif handedness == "Right" and swinging:
                dire_label="Right" if dire>0 else "Left"

            # ── 观察点D：拇指判断 ──
            print(f"[D] 拇指: lm[4].x={lm[4][0]:.3f}  "
                f"lm[3].x={lm[3][0]:.3f}  "
                f"伸出={thumb_up}")
            fingers[0] = thumb_up
            if thumb_up:
                finger_count += 1

            #判断手势
            gesture = "unknown"
            for name, pattern in GESTURES.items():
                match = all(p is None or fingers[i]==p for i,p in enumerate(pattern))
                if match:
                    gesture = name
                    break
            history.append(finger_count)
            stable_count = round(sum(history) / len(history))
            # ── 观察点E：最终结果 ──
            print(f"[E] ★ 最终手指数 = {finger_count}")
           
            print(f"speed={speed:.4f}")
            print(f"[F] fingers = {fingers}")
            print("-" * 40)

            
        cv2.putText(frame, str(stable_count),
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
        label = "SWING!" if swinging else "static"
        cv2.putText(frame, label, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
        cv2.putText(frame, dire_label, (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
        cv2.putText(frame, gesture, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
        cv2.imshow("Finger Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # 点击右上角 X 退出
        if cv2.getWindowProperty("Finger Counter", cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()