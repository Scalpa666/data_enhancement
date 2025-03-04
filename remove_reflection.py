import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# 输入文件夹和输出文件夹路径
input_folder = "data/image2"
output_folder = "data/image2_rr_y"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有图片
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 可以根据需要调整文件类型
        # 构建完整的路径
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 读取图像
        image = cv2.imread(input_path)
        if image is None:
            print(f"无法读取图像: {input_path}")
            continue

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 提取明度通道（V）
        v = hsv[:, :, 2]

        # 定义高光阈值（例如：亮度 > 200）
        v_threshold = 200
        v_mask = cv2.threshold(v, v_threshold, 255, cv2.THRESH_BINARY)[1]

        # 提取Blue通道（B）
        b = image[:, :, 0]

        # 定义B通道阈值排除黄色
        b_threshold = 150
        b_mask = cv2.threshold(b, b_threshold, 255, cv2.THRESH_BINARY)[1]

        # 反光区域且非黄色
        combined_mask = cv2.bitwise_and(v_mask, b_mask)

        # 利用mask对高光区域处理
        # 降低明度（例如：降低到原值的70%）
        hsv[:, :, 2] = np.where(combined_mask, hsv[:, :, 2] * 0.9, hsv[:, :, 2])

        # 降低饱和度（例如：降低到原值的80%）
        # hsv[:, :, 1] = np.where(v_mask, hsv[:, :, 1] * 0.8, hsv[:, :, 1])

        # 对掩膜进行高斯模糊（平滑边缘）
        # v_mask_blur = cv2.GaussianBlur(v_mask, (15, 15), 0)
        # v_mask_blur = v_mask_blur / 255.0  # 归一化到[0,1]
        # 渐进式调整明度和饱和度
        # hsv[:, :, 2] = (hsv[:, :, 2] * (1 - 0.3 * v_mask_blur)).astype(np.uint8)
        # hsv[:, :, 1] = (hsv[:, :, 1] * (1 - 0.2 * v_mask_blur)).astype(np.uint8)

        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 保存结果图像
        cv2.imwrite(output_path, result)
        print(f"已处理并保存: {output_path}")

        # Display the original and enhanced images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Processed Image")
        plt.imshow(result)
        plt.axis('off')

        plt.show()

# 批量处理完成
print("所有图片处理完成！")