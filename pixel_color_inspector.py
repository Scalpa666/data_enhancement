import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_pixel_info(event, x, y, flags, param):
    # 当鼠标左键按下时触发
    if event == cv2.EVENT_LBUTTONDOWN:
        # 获取BGR值（OpenCV默认格式）
        bgr = image[y, x]
        # 转换为RGB
        rgb = bgr[::-1]
        # 转换为HSV
        hsv = cv2.cvtColor(np.array([[bgr]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]

        # 输出信息
        print(f"像素坐标: ({x}, {y})")
        print(f"RGB值: {tuple(rgb)}")
        print(f"HSV值: {tuple(hsv)}")
        print("-" * 30)


if __name__ == "__main__":
    # 读取图像（替换为你的图像路径）
    image_path = "data/09_frame_0001.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        exit()

    # 创建窗口并设置回调函数
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", get_pixel_info)

    # 显示图像
    while True:
        cv2.imshow("Image", image)
        # 按ESC键退出
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()