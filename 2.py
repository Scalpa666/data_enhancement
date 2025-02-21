import os
import json
from PIL import Image
import math


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
        # 转换为新的坐标系统
        # 这里需要根据旋转角度重新计算边界框坐标
        # 这里只是一种简单的示例计算方法，实际情况可能更复杂
        # 假设旋转后的边界框中心不变，但需要重新计算角度
        # 这里仅作示例，实际可能需要更复杂的计算
        # 这里我们暂时不进行复杂的坐标变换，直接保存原边界框
        # 用户可根据实际情况调整
        # 这里使用一个简单变换，边界框围绕中心旋转
        # 获取原始边界框的中心坐标
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        # 计算旋转后的边界框
        # 这里仅作为示例，实际可能需要更复杂的计算
        # 假设边界框大小不变，位置旋转
        # 这里使用一个简单的矩阵变换
        # 这是一个简化的实现，实际坐标变换可能更复杂
        # 计算旋转后的边界框中心点坐标
        # 使用旋转矩阵：
        # [cosθ, -sinθ]
        # [sinθ, cosθ]
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
        annotation['bbox'] = [new_xmin, new_ymin, new_xmax, new_ymax]

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

    # 更新标签中的边界框
    for annotation in label_data['shapes']:
        xmin, ymin = annotation['points'][0]
        xmax, ymax = annotation['points'][1]
        width, height = image.size
        if direction == 'horizontal':
            # 水平翻转：xmin和xmax变为宽度-原值
            new_xmin = width - xmax
            new_xmax = width - xmin
            new_ymin = ymin
            new_ymax = ymax
        elif direction == 'vertical':
            # 垂直翻转：ymin和ymax变为高度-原值
            new_xmin = xmin
            new_xmax = xmax
            new_ymin = height - ymax
            new_ymax = height - ymin
        else:
            new_xmin, new_ymin, new_xmax, new_ymax = xmin, ymin, xmax, ymax
        annotation['bbox'] = [new_xmin, new_ymin, new_xmax, new_ymax]

    # 保存翻转后的图像和标签
    output_image_path = os.path.join(output_image_dir,
                                     os.path.basename(image_path).replace('.jpg', f'_flip_{direction}.jpg'))
    flipped_image.save(output_image_path)
    output_label_path = os.path.join(output_label_dir,
                                     os.path.basename(label_path).replace('.json', f'_flip_{direction}.json'))
    with open(output_label_path, 'w') as f:
        json.dump(label_data, f, indent=4)


def mirror_image_and_boxes(image_path, label_path, output_image_dir, output_label_dir):
    """
    镜像图像并更新标签中的边界框
    :param image_path: 输入图像路径
    :param label_path: 输入标签路径
    :param output_image_dir: 输出图像路径
    :param output_label_dir: 输出标签路径
    """
    # 加载图像和标签
    image = Image.open(image_path)
    with open(label_path, 'r') as f:
        label_data = json.load(f)

    # 镜像图像（假设镜像是水平镜像）
    mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 更新标签中的边界框
    for annotation in label_data['shapes']:
        xmin, ymin = annotation['points'][0]
        xmax, ymax = annotation['points'][1]
        width, _ = image.size
        new_xmin = width - xmax
        new_xmax = width - xmin
        annotation['bbox'] = [new_xmin, ymin, new_xmax, ymax]

    # 保存镜像后的图像和标签
    output_image_path = os.path.join(output_image_dir, os.path.basename(image_path).replace('.jpg', '_mirror.jpg'))
    mirrored_image.save(output_image_path)
    output_label_path = os.path.join(output_label_dir, os.path.basename(label_path).replace('.json', '_mirror.json'))
    with open(output_label_path, 'w') as f:
        json.dump(label_data, f, indent=4)


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
            rotate_image_and_boxes(image_path, label_path, 30, output_image_dir, output_label_dir)
            rotate_image_and_boxes(image_path, label_path, 60, output_image_dir, output_label_dir)

            # 水平翻转
            flip_image_and_boxes(image_path, label_path, 'horizontal', output_image_dir, output_label_dir)

            # 垂直翻转
            flip_image_and_boxes(image_path, label_path, 'vertical', output_image_dir, output_label_dir)

            # 镜像
            mirror_image_and_boxes(image_path, label_path, output_image_dir, output_label_dir)

            print(f"Processed {filename}")


# 示例使用
image_dir = 'test/image'
label_dir = 'test/label'
output_image_dir = 'test/image_augmented'
output_label_dir = 'test/label_augmented'
generate_augmented_dataset(image_dir, label_dir, output_image_dir, output_label_dir)