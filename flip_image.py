import os
import json
from PIL import Image
import math

class_mapping = {
    'slight': 0,
    'serious': 1,
    'healthy': 2,
    'old': 3
}
def rotate_image_and_boxes(image_path, label_path, angle, output_image_dir, output_label_dir):
    """
    旋转图像并更新标签中的边界框
    :param image_path: 输入图像路径
    :param label_path: 输入标签路径
    :param angle: 旋转角度
    :param output_image_dir: 输出图像路径
    :param output_label_dir: 输出标签路径
    """
    # 加载图像和标签
    image = Image.open(image_path)
    with open(label_path, 'r') as f:
        label_data = json.load(f)

    # 旋转图像
    rotated_image = image.rotate(angle, expand=True)

    # 计算新的图像宽高
    new_width, new_height = rotated_image.size

    # 更新标签中的边界框
    for annotation in label_data['shapes']:
        # 获取原始边界框
        xmin, ymin = annotation['points'][0]
        xmax, ymax = annotation['points'][1]

        # 获取原始边界框的中心坐标
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        theta = angle * 3.1416 / 180  # 转换为弧度
        cos_theta = round(math.cos(theta), 2)
        sin_theta = round(math.sin(theta), 2)
        # 计算新的中心坐标
        new_center_x = int((center_x * cos_theta - center_y * sin_theta) + (new_width / 2))
        new_center_y = int((center_x * sin_theta + center_y * cos_theta) + (new_height / 2))
        # 重新计算边界框的宽高
        # 这里假设边界框宽高不变
        width = xmax - xmin
        height = ymax - ymin
        # 新的边界框
        new_xmin = new_center_x - width / 2
        new_ymin = new_center_y - height / 2
        new_xmax = new_center_x + width / 2
        new_ymax = new_center_y + height / 2
        # 调整边界框确保不超出图像边界
        new_xmin = max(0, new_xmin)
        new_ymin = max(0, new_ymin)
        new_xmax = min(new_width, new_xmax)
        new_ymax = min(new_height, new_ymax)
        annotation['points'] = [[new_xmin, new_ymin], [new_xmax, new_ymax]]

    # 保存旋转后的图像和标签
    output_image_path = os.path.join(output_image_dir,
                                     os.path.basename(image_path).replace('.jpg', f'_rotate_{angle}.jpg'))
    rotated_image.save(output_image_path)
    output_label_path = os.path.join(output_label_dir,
                                     os.path.basename(label_path).replace('.json', f'_rotate_{angle}.json'))
    with open(output_label_path, 'w') as f:
        json.dump(label_data, f, indent=4)


def flip_image_and_boxes(image_path, label_path, direction, output_image_dir, output_label_dir):
    """
    翻转图像并更新标签中的边界框
    :param image_path: 输入图像路径
    :param label_path: 输入标签路径
    :param direction: 翻转方向（horizontal或vertical）
    :param output_image_dir: 输出图像路径
    :param output_label_dir: 输出标签路径
    """
    # 加载图像和标签
    image = Image.open(image_path)
    with open(label_path, 'r') as f:
        label_data = json.load(f)

    # 翻转图像
    if direction == 'horizontal':
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif direction == 'vertical':
        flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        raise ValueError("Invalid flip direction. Use 'horizontal' or 'vertical'.")


    # 保存翻转后的图像
    output_image_path = os.path.join(output_image_dir,
                                     os.path.basename(image_path).replace('.jpg', f'_flip_{direction}.jpg'))
    flipped_image.save(output_image_path)

    # 保存标签为 txt 文件
    output_label_path = os.path.join(output_label_dir,
                                     os.path.basename(label_path).replace('.json', f'_flip_{direction}.txt'))
    with open(output_label_path, 'w') as f:
        # 写入每个边界框信息
        for annotation in label_data['shapes']:
            class_name = annotation['label']
            if class_name not in class_mapping:
                print(f"Warning: Unknown class {class_name}")
                continue
            class_id = class_mapping[class_name]
            xmin, ymin = annotation['points'][0]
            xmax, ymax = annotation['points'][1]
            width, height = image.size
            if direction == 'horizontal':
                new_xmin = width - xmax
                new_xmax = width - xmin
                new_ymin = ymin
                new_ymax = ymax
            elif direction == 'vertical':
                new_xmin = xmin
                new_xmax = xmax
                new_ymin = height - ymax
                new_ymax = height - ymin

            x_center = (new_xmin + new_xmax) / 2.0 / width
            y_center = (new_ymin + new_ymax) / 2.0 / height
            bbox_width = (new_xmax - new_xmin) / width
            bbox_height = (new_ymax - new_ymin) / height

            # 格式化变量
            x_center = "{:.6f}".format(x_center)
            y_center = "{:.6f}".format(y_center)
            bbox_width = "{:.6f}".format(bbox_width)
            bbox_height = "{:.6f}".format(bbox_height)

            f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

def generate_augmented_dataset(image_dir, label_dir, output_image_dir, output_label_dir):
    """
    生成增强后的数据集
    :param image_dir: 输入图像目录
    :param label_dir: 输入标签目录
    :param output_image_dir: 输出图像目录
    :param output_label_dir: 输出标签目录
    """
    # 创建输出目录
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 遍历图像文件夹中的所有图像
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            # 构建输入路径
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace('.jpg', '.json'))

            # 检查标签文件是否存在
            if not os.path.exists(label_path):
                print(f"Warning: Label file not found for {filename}")
                continue

            # 增强操作
            # 旋转
            # rotate_image_and_boxes(image_path, label_path, 30, output_image_dir, output_label_dir)
            # rotate_image_and_boxes(image_path, label_path, 60, output_image_dir, output_label_dir)

            # 水平翻转
            flip_image_and_boxes(image_path, label_path, 'horizontal', output_image_dir, output_label_dir)

            # 垂直翻转
            flip_image_and_boxes(image_path, label_path, 'vertical', output_image_dir, output_label_dir)

            print(f"Processed {filename}")


# 示例使用
image_dir = 'data/image'
label_dir = 'data/label'
output_image_dir = 'data/image_augmented'
output_label_dir = 'data/label_augmented'
generate_augmented_dataset(image_dir, label_dir, output_image_dir, output_label_dir)