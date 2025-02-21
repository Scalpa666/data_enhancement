import os
import json
import cv2
import copy
import numpy as np
from tqdm import tqdm


class DataAugmentor:
    def __init__(self, image_dir, json_path, output_dir):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.augmented_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

        # 加载原始标注数据
        with open(json_path) as f:
            self.original_data = json.load(f)

        self.augmented_data["categories"] = self.original_data["categories"]

    def horizontal_flip(self, image, bboxes):
        """水平翻转"""
        h, w = image.shape[:2]
        flipped = cv2.flip(image, 1)

        new_bboxes = []
        for bbox in bboxes:
            x, y, width, height = bbox
            new_x = w - x - width
            new_bbox = [new_x, y, width, height]
            new_bboxes.append(new_bbox)

        return flipped, new_bboxes

    def vertical_flip(self, image, bboxes):
        """垂直翻转"""
        h, w = image.shape[:2]
        flipped = cv2.flip(image, 0)

        new_bboxes = []
        for bbox in bboxes:
            x, y, width, height = bbox
            new_y = h - y - height
            new_bbox = [x, new_y, width, height]
            new_bboxes.append(new_bbox)

        return flipped, new_bboxes

    def process(self):
        # 复制原始数据
        self.augmented_data["images"] = copy.deepcopy(self.original_data["images"])
        self.augmented_data["annotations"] = copy.deepcopy(self.original_data["annotations"])

        # 遍历所有图像
        for img_info in tqdm(self.original_data["images"]):
            img_id = img_info["id"]
            file_name = img_info["file_name"]
            width = img_info["width"]
            height = img_info["height"]

            # 读取图像
            img_path = os.path.join(self.image_dir, file_name)
            image = cv2.imread(img_path)

            # 获取对应标注
            annotations = [anno for anno in self.original_data["annotations"]
                           if anno["image_id"] == img_id]
            bboxes = [anno["bbox"] for anno in annotations]

            # 生成增强数据
            self._apply_augmentation(image, bboxes, img_info, "horizontal")
            self._apply_augmentation(image, bboxes, img_info, "vertical")

        # 保存新标注文件
        output_json = os.path.join(self.output_dir, "augmented_annotations.json")
        with open(output_json, "w") as f:
            json.dump(self.augmented_data, f)

    def _apply_augmentation(self, original_img, original_bboxes, original_img_info, flip_type):
        # 生成唯一ID
        new_img_id = len(self.augmented_data["images"]) + 1
        new_anno_id = len(self.augmented_data["annotations"]) + 1

        # 执行翻转
        if flip_type == "horizontal":
            img, bboxes = self.horizontal_flip(original_img, original_bboxes)
            suffix = "_hflip"
        elif flip_type == "vertical":
            img, bboxes = self.vertical_flip(original_img, original_bboxes)
            suffix = "_vflip"
        else:
            return

        # 保存新图像
        original_name = os.path.splitext(original_img_info["file_name"])[0]
        new_filename = f"{original_name}{suffix}.jpg"
        cv2.imwrite(os.path.join(self.output_dir, "images", new_filename), img)

        # 创建新图像记录
        new_img_info = {
            "id": new_img_id,
            "file_name": new_filename,
            "width": original_img_info["width"],
            "height": original_img_info["height"]
        }
        self.augmented_data["images"].append(new_img_info)

        # 创建新标注记录
        for i, bbox in enumerate(bboxes):
            new_anno = {
                "id": new_anno_id + i,
                "image_id": new_img_id,
                "category_id": original_bboxes[i]["category_id"],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            }
            self.augmented_data["annotations"].append(new_anno)


if __name__ == "__main__":
    # 配置参数
    IMAGE_DIR = "path/to/original/images"
    JSON_PATH = "path/to/annotations.json"
    OUTPUT_DIR = "path/to/augmented_dataset"

    # 执行增强
    augmentor = DataAugmentor(IMAGE_DIR, JSON_PATH, OUTPUT_DIR)
    augmentor.process()
    print("数据增强完成！")