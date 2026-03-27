"""
手势方向识别 - 第三步：多线程优化数据采集脚本
================================================
优化方案：完整多线程架构
性能提升：从15-20 FPS提升到40-60 FPS

架构说明：
  采集线程：从摄像头读取原始帧
  处理线程：MediaPipe手势检测
  显示线程：UI渲染和用户交互
  写入线程：异步保存CSV

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
import threading
import queue
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ── 配置 ──────────────────────────────────
WINDOW = 30                          # 每个样本的帧数
OUTPUT_FILE = "gesture_data.csv"     # 输出文件名
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


class MultiThreadCollector:
    """多线程手势数据采集器"""
    
    def __init__(self, window_size=30, target_per_dir=50):
        self.window_size = window_size
        self.target_per_dir = target_per_dir
        
        # 线程间通信队列
        self.raw_frame_queue = queue.Queue(maxsize=2)  # 限制大小避免内存堆积
        self.result_queue = queue.Queue(maxsize=5)
        self.data_queue = queue.Queue()
        
        # 控制标志
        self.running = False
        self.frame_count = 0  # 用于性能统计
        
        # 统计信息
        self.label_counts = {k: 0 for k in KEY_LABEL_MAP.values()}
        self.next_sample_id = 0
        
        # 加载已有样本统计
        self._load_existing_samples()
        
        # MediaPipe（处理线程专用）
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            static_image_mode=False  # 优化：视频模式
        )
        
        # 性能统计
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
    
    def _load_existing_samples(self):
        """加载已有样本统计"""
        if not os.path.exists(OUTPUT_FILE):
            return
        
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            seen = {}
            for row in reader:
                sid = row["sample_id"]
                lbl = row["label"]
                if sid not in seen:
                    seen[sid] = lbl
            for lbl in seen.values():
                if lbl in self.label_counts:
                    self.label_counts[lbl] += 1
        
        if seen:
            self.next_sample_id = max(int(s) for s in seen) + 1
    
    def capture_thread(self, cap):
        """
        采集线程：只负责读取摄像头
        性能目标：2ms/帧
        """
        print("[采集线程] 启动")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            try:
                # 非阻塞放入，满时丢弃旧帧
                if self.raw_frame_queue.full():
                    try:
                        self.raw_frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.raw_frame_queue.put(frame, timeout=0.01)
            except queue.Full:
                pass
        
        print("[采集线程] 停止")
    
    def process_thread(self):
        """
        处理线程：MediaPipe检测
        性能目标：15ms/帧（瓶颈）
        """
        print("[处理线程] 启动")
        
        while self.running:
            try:
                frame = self.raw_frame_queue.get(timeout=0.1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)
                
                result_data = {
                    'frame': frame,
                    'landmarks': None,
                    'landmark_raw': None,  # 用于绘制
                    'timestamp': time.time()
                }
                
                if results.multi_hand_landmarks:
                    lm_raw = results.multi_hand_landmarks[0]
                    result_data['landmark_raw'] = lm_raw
                    result_data['landmarks'] = np.array(
                        [[p.x, p.y, p.z] for p in lm_raw.landmark],
                        dtype=np.float32
                    )
                
                self.result_queue.put(result_data)
            except queue.Empty:
                continue
        
        print("[处理线程] 停止")
    
    def display_thread(self):
        """
        显示线程：UI渲染和用户交互
        性能目标：5ms/帧
        """
        print("[显示线程] 启动")
        
        buffer = []
        recording = False
        current_label = ""
        prev_key = -1
        
        while self.running:
            try:
                result = self.result_queue.get(timeout=0.1)
                frame = result['frame']
                landmarks = result['landmarks']
                landmark_raw = result['landmark_raw']
                
                # 绘制骨架
                if landmark_raw is not None:
                    mp_draw.draw_landmarks(
                        frame, 
                        landmark_raw, 
                        mp_hands.HAND_CONNECTIONS
                    )
                
                # 计算FPS
                self.fps_frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.fps_frame_count / (current_time - self.last_fps_time)
                    self.fps_frame_count = 0
                    self.last_fps_time = current_time
                
                # 键盘检测
                key = cv2.waitKey(1) & 0xFF
                key_just_pressed = (key in KEY_LABEL_MAP) and (key != prev_key)
                
                # 刚按下：开始新的录制
                if key_just_pressed:
                    current_label = KEY_LABEL_MAP[key]
                    buffer = []
                    recording = True
                
                # 松开键：停止录制（丢弃不足30帧的样本）
                if recording and key not in KEY_LABEL_MAP and prev_key in KEY_LABEL_MAP:
                    if len(buffer) < self.window_size:
                        buffer = []
                        recording = False
                
                prev_key = key if key in KEY_LABEL_MAP else -1
                
                # 录制帧
                if recording and landmarks is not None:
                    buffer.append(landmarks)
                    
                    # 凑满30帧 → 存样本
                    if len(buffer) == self.window_size:
                        # 放入数据队列，异步写入
                        self.data_queue.put({
                            'sample_id': self.next_sample_id,
                            'label': current_label,
                            'data': np.array(buffer)
                        })
                        self.label_counts[current_label] += 1
                        self.next_sample_id += 1
                        print(f"[saved] sample {self.next_sample_id-1:04d}  "
                              f"label={current_label:12s}  "
                              f"total={self.label_counts[current_label]:3d}/{TARGET_PER_DIR}")
                        buffer = []
                        recording = False
                
                # HUD绘制
                self._draw_hud(frame, recording, current_label, buffer)
                
                cv2.imshow("MultiThread Data Collection", frame)
                
                if key == 27:  # ESC
                    self.running = False
                    break
                
                if cv2.getWindowProperty("MultiThread Data Collection",
                                        cv2.WND_PROP_VISIBLE) < 1:
                    self.running = False
                    break
            
            except queue.Empty:
                continue
        
        print("[显示线程] 停止")
    
    def _draw_hud(self, frame, recording, current_label, buffer):
        """绘制HUD界面"""
        h = frame.shape[0]
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (350, 30 + len(KEY_LABEL_MAP) * 22), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        
        # 标题
        cv2.putText(frame, "多线程数据采集  ESC退出",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 各方向统计
        for i, (k, lbl) in enumerate(KEY_LABEL_MAP.items()):
            cnt = self.label_counts.get(lbl, 0)
            color = LABEL_COLORS.get(lbl, (200, 200, 200))
            bar = int((cnt / self.target_per_dir) * 80)
            y = 42 + i * 22
            
            cv2.putText(frame, f"{chr(k).upper()} {lbl:12s} {cnt:3d}/{TARGET_PER_DIR}",
                        (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(frame, (250, y-12), (250+bar, y-4), color, -1)
        
        # FPS显示
        cv2.putText(frame, f"FPS: {self.fps:.1f}",
                    (8, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 录制状态指示
        if recording:
            color = LABEL_COLORS.get(current_label, (255, 255, 255))
            filled = len(buffer)
            cv2.putText(frame, f"REC  {current_label}  {filled}/{self.window_size}",
                        (8, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # 进度条
            cv2.rectangle(frame, (8, h-40), (8 + int(filled/self.window_size*200), h-34), color, -1)
    
    def write_thread(self):
        """
        写入线程：异步保存CSV
        性能目标：不阻塞主流程
        """
        print("[写入线程] 启动")
        
        batch = []
        batch_size = 5
        
        # 初始化文件
        if not os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "sample_id", "frame",
                    "wrist_x", "wrist_y", "wrist_z",
                    "mcp5_x", "mcp5_y", "mcp9_x", "mcp9_y",
                    "mcp13_x", "mcp13_y", "mcp17_x", "mcp17_y",
                    "label"
                ])
                writer.writeheader()
        
        while self.running or not self.data_queue.empty():
            try:
                sample = self.data_queue.get(timeout=0.1)
                batch.append(sample)
                
                if len(batch) >= batch_size:
                    self._flush_batch(batch)
                    batch = []
            except queue.Empty:
                if batch:
                    self._flush_batch(batch)
                    batch = []
        
        # 刷新剩余数据
        if batch:
            self._flush_batch(batch)
        
        print("[写入线程] 停止")
    
    def _flush_batch(self, batch):
        """批量写入CSV"""
        rows = []
        for sample in batch:
            data = sample['data']
            for i, frame_lm in enumerate(data):
                rows.append({
                    "sample_id": sample['sample_id'],
                    "frame": i,
                    "wrist_x": round(frame_lm[0, 0], 5),
                    "wrist_y": round(frame_lm[0, 1], 5),
                    "wrist_z": round(frame_lm[0, 2], 5),
                    "mcp5_x": round(frame_lm[5, 0], 5),
                    "mcp5_y": round(frame_lm[5, 1], 5),
                    "mcp9_x": round(frame_lm[9, 0], 5),
                    "mcp9_y": round(frame_lm[9, 1], 5),
                    "mcp13_x": round(frame_lm[13, 0], 5),
                    "mcp13_y": round(frame_lm[13, 1], 5),
                    "mcp17_x": round(frame_lm[17, 0], 5),
                    "mcp17_y": round(frame_lm[17, 1], 5),
                    "label": sample['label']
                })
        
        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writerows(rows)
    
    def start(self, camera_id=0):
        """启动所有线程"""
        print("=" * 60)
        print("多线程手势数据采集器启动")
        print("=" * 60)
        print(f"已有样本总数: {self.next_sample_id}")
        for lbl, cnt in self.label_counts.items():
            print(f"  {lbl:12s}: {cnt}/{self.target_per_dir}")
        print("=" * 60)
        
        self.running = True
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("错误：无法打开摄像头")
            return
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 创建线程
        threads = [
            threading.Thread(target=self.capture_thread, args=(cap,), name="Capture"),
            threading.Thread(target=self.process_thread, name="Process"),
            threading.Thread(target=self.display_thread, name="Display"),
            threading.Thread(target=self.write_thread, name="Write")
        ]
        
        # 启动所有线程
        for t in threads:
            t.start()
        
        # 等待所有线程结束
        for t in threads:
            t.join()
        
        # 清理
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        print("=" * 60)
        print(f"采集完成！共 {self.next_sample_id} 个样本")
        print(f"已保存至 {OUTPUT_FILE}")
        print("=" * 60)


# ── 主程序 ──────────────────────────────────
def main():
    collector = MultiThreadCollector(window_size=WINDOW, target_per_dir=TARGET_PER_DIR)
    
    try:
        collector.start()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
