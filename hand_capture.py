import cv2
import mediapipe as mp

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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        finger_count = 0

        if results.multi_hand_landmarks:
            lm_list = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label

            # ── 观察点A：左右手标签 ──
            print(f"[A] 检测到：{handedness}")

            # ① 提取坐标
            lm = []
            for i in lm_list.landmark:
                lm.append((i.x, i.y, i.z))

            # ── 观察点B：手腕和食指尖的坐标 ──
            print(f"[B] 手腕    lm[0]  = {lm[0]}")
            print(f"[B] 食指根  lm[5]  = {lm[5]}")
            print(f"[B] 食指尖  lm[8]  = {lm[8]}")

            # ② 循环判断四根手指
            finger_count = 0
            THRESHOLD = 0.05
            for name, (mcp_i, tip_i) in FINGER_PAIRS.items():
                is_up = (lm[mcp_i][1] - lm[tip_i][1])>THRESHOLD

                # ── 观察点C：每根手指的判断过程 ──
                print(f"[C] {name}: mcp_y={lm[mcp_i][1]:.3f}  "
                    f"tip_y={lm[tip_i][1]:.3f}  "
                    f"伸出={is_up}")

                if is_up:
                    finger_count += 1
            thumb=0.028
            # ③ 拇指,以摄像头为出发点
            if handedness == "Left":
                thumb_up = (lm[3][0] - lm[4][0])>thumb
                print(f"[D详细] handedness={handedness}")
                print(f"[D详细] lm[4].x={lm[4][0]:.3f}  lm[3].x={lm[3][0]:.3f}")
                print(f"[D详细] 公式结果: {lm[3][0] - lm[4][0]:.3f}  THRESHOLD={thumb}")
                print(f"[D详细] thumb_up={thumb_up}")
            else:
                thumb_up = (lm[3][0] - lm[4][0])<thumb
                print(f"[D详细] handedness={handedness}")
                print(f"[D详细] lm[4].x={lm[4][0]:.3f}  lm[3].x={lm[3][0]:.3f}")
                print(f"[D详细] 公式结果: {lm[3][0] - lm[4][0]:.3f}  THRESHOLD={thumb}")
                print(f"[D详细] thumb_up={thumb_up}")
            # ── 观察点D：拇指判断 ──
            print(f"[D] 拇指: lm[4].x={lm[4][0]:.3f}  "
                f"lm[3].x={lm[3][0]:.3f}  "
                f"伸出={thumb_up}")

            if thumb_up:
                finger_count += 1

            # ── 观察点E：最终结果 ──
            print(f"[E] ★ 最终手指数 = {finger_count}")
            print("-" * 40)

        cv2.putText(frame, str(finger_count),
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
        cv2.imshow("Finger Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # 点击右上角 X 退出
        if cv2.getWindowProperty("Finger Counter", cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()