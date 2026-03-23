#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv; // 遵循你的习惯，不使用std::
using namespace std;

int main() {
    // 1. 打开摄像头（0是默认索引）
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "无法打开摄像头！" << endl;
        return -1;
    }

    Mat frame, gray, flipped;
    namedWindow("Action Insight", WINDOW_AUTOSIZE);

    while (true) {
        cap >> frame; // 读取当前帧
        if (frame.empty()) break;

        // 2. 预处理：动作识别通常不需要色彩，灰度化能显著降低计算量
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 3. 镜像翻转：让操作更符合人的视觉直觉
        flip(gray, flipped, 1);

        imshow("Action Insight", flipped);

        if (waitKey(30) == 27) break; // 按下ESC退出
    }
    return 0;
}