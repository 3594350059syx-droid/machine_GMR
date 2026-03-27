"""
手势方向识别 - 第二步：数据采集脚本
=====================================
操作说明：
  按住对应键开始录制，松开停止
  A=left  D=right  W=up  S=down
  Q=up-left  E=up-right  Z=down-left  C=down-right
  ESC 退出并保存

采集目标：每个方向录50次，共400个样本
"""

import cv2
import csv
import os
import mediapipe as mp
from collections import deque

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# ── 配置 ──────────────────────────────────
WINDOW       = 30                    # 每个样本的帧数
OUTPUT_FILE  = "gesture_data.csv"   # 输出文件名
TARGET_PER_DIR = 50                  # 每个方向目标采集次数

KEY_LABEL_MAP = {
    ord('d'): "right",
    ord('a'): "left",
    ord('w'): "up",
    ord('s'): "down",
    ord('e'): "up-right",
    ord('q'): "up-left",
    ord('c'): "down-right",
    ord('z'): "down-left",
}

LABEL_COLORS = {
    "right":      (255, 100, 100),
    "left":       (100, 100, 255),
    "up":         (100, 255, 100),
    "down":       (255, 255, 100),
    "up-right":   (255, 180, 100),
    "up-left":    (180, 255, 100),
    "down-right": (255, 100, 180),
    "down-left":  (100, 255, 255),
}

# ── CSV 写入（追加模式，每样本立即写入）──
def save_sample(rows, filename=OUTPUT_FILE):
    file_exists = os.path.exists(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "frame",
            "wrist_x", "wrist_y", "wrist_z",
            "mcp5_x",  "mcp5_y",
            "mcp9_x",  "mcp9_y",
            "mcp13_x", "mcp13_y",
            "mcp17_x", "mcp17_y",
            "label"
        ])
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

# ── 统计已有样本数 ──
def count_existing(filename=OUTPUT_FILE):
    counts = {k: 0 for k in KEY_LABEL_MAP.values()}
    if not os.path.exists(filename):
        return counts, 0
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        seen = {}
        for row in reader:
            sid = row["sample_id"]
            lbl = row["label"]
            if sid not in seen:
                seen[sid] = lbl
        for lbl in seen.values():
            if lbl in counts:
                counts[lbl] += 1
    total = max((int(s) for s in seen), default=-1) + 1
    return counts, total

# ── 主程序 ──────────────────────────────────
def main():
    label_counts, next_sample_id = count_existing()
    print(f"已有样本总数: {next_sample_id}")
    for lbl, cnt in label_counts.items():
        print(f"  {lbl:12s}: {cnt}/{TARGET_PER_DIR}")

    cap = cv2.VideoCapture(0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buffer        = []
    recording     = False
    current_label = ""
    prev_key      = -1    # 上一帧按键，用于检测"刚按下"

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            lm = None
            if results.multi_hand_landmarks:
                lm_raw = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, lm_raw, mp_hands.HAND_CONNECTIONS)
                lm = [(p.x, p.y, p.z) for p in lm_raw.landmark]

            # ── 键盘检测 ──
            key = cv2.waitKey(1) & 0xFF
            key_just_pressed = (key in KEY_LABEL_MAP) and (key != prev_key)

            # 刚按下：开始新的录制
            if key_just_pressed:
                current_label = KEY_LABEL_MAP[key]
                buffer        = []
                recording     = True

            # 松开键：停止录制（丢弃不足30帧的样本）
            if recording and key not in KEY_LABEL_MAP and prev_key in KEY_LABEL_MAP:
                if len(buffer) < WINDOW:
                    buffer    = []
                    recording = False

            prev_key = key if key in KEY_LABEL_MAP else -1

            # ── 录制帧 ──
            if recording and lm is not None:
                buffer.append(lm)

                # 凑满30帧 → 存样本
                if len(buffer) == WINDOW:
                    rows = []
                    for i, frame_lm in enumerate(buffer):
                        rows.append({
                            "sample_id": next_sample_id,
                            "frame":     i,
                            "wrist_x":   round(frame_lm[0][0], 5),
                            "wrist_y":   round(frame_lm[0][1], 5),
                            "wrist_z":   round(frame_lm[0][2], 5),
                            "mcp5_x":    round(frame_lm[5][0], 5),
                            "mcp5_y":    round(frame_lm[5][1], 5),
                            "mcp9_x":    round(frame_lm[9][0], 5),
                            "mcp9_y":    round(frame_lm[9][1], 5),
                            "mcp13_x":   round(frame_lm[13][0], 5),
                            "mcp13_y":   round(frame_lm[13][1], 5),
                            "mcp17_x":   round(frame_lm[17][0], 5),
                            "mcp17_y":   round(frame_lm[17][1], 5),
                            "label":     current_label,
                        })
                    save_sample(rows)
                    label_counts[current_label] += 1
                    next_sample_id += 1
                    print(f"[saved] sample {next_sample_id-1:04d}  label={current_label}  "
                          f"total {current_label}={label_counts[current_label]}")
                    buffer    = []
                    recording = False

            # ── HUD ──
            h = frame.shape[0]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (320, 30 + len(KEY_LABEL_MAP) * 22), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

            cv2.putText(frame, "按键录制方向样本  ESC退出",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            for i, (k, lbl) in enumerate(KEY_LABEL_MAP.items()):
                cnt   = label_counts.get(lbl, 0)
                color = LABEL_COLORS.get(lbl, (200,200,200))
                bar   = int((cnt / TARGET_PER_DIR) * 80)
                y     = 42 + i * 22
                cv2.putText(frame, f"{chr(k).upper()} {lbl:12s} {cnt:3d}/{TARGET_PER_DIR}",
                            (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.rectangle(frame, (220, y-12), (220+bar, y-4), color, -1)

            # 录制状态指示
            if recording:
                color  = LABEL_COLORS.get(current_label, (255,255,255))
                filled = len(buffer)
                cv2.putText(frame, f"REC  {current_label}  {filled}/{WINDOW}",
                            (8, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                # 进度条
                cv2.rectangle(frame, (8, h-10), (8 + int(filled/WINDOW*200), h-4), color, -1)

            cv2.imshow("Data Collection - Step 2", frame)

            if key == 27:   # ESC
                break
            if cv2.getWindowProperty("Data Collection - Step 2",
                                      cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n采集完成！共 {next_sample_id} 个样本，已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
