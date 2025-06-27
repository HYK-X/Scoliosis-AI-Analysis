# 文件名: yolo_detector.py
# -*- coding: utf-8 -*-

from ultralytics import YOLO
import torch
import numpy as np


def extract_keypoints_from_image(model_path, image_path, conf_threshold=0.25):
    """
    加载模型并对单张图片进行预测，提取所有检测到的关键点坐标。
    如果检测到的实例超过17个，则只保留置信度最高的17个。

    Args:
        model_path (str): 训练好的模型权重路径 (.pt文件)。
        image_path (str): 待预测的图片路径。
        conf_threshold (float): 初始置信度阈值。

    Returns:
        tuple: (keypoints, original_image)
            - keypoints (np.ndarray or None): 形状为 (M, 2) 的Numpy数组，M是总点数。
            - original_image (np.ndarray or None): OpenCV格式的原始图像(BGR)。
    """
    TARGET_VERTEBRAE_COUNT = 17

    print("--- [模块A, 步骤1: 加载模型并预测] ---")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"!! 错误: 加载模型时出错: {e}")
        return None, None

    results = model(image_path, conf=conf_threshold)
    print("预测完成。")

    if not results:
        print("!! 预测结果为空。")
        return None, None

    result = results[0]
    original_image = result.orig_img  # 提取原始图像，以便后续使用

    if result.keypoints is None or result.keypoints.shape[1] == 0:
        print("!! 在图片上没有检测到任何关键点。")
        return None, original_image
    print(result.keypoints)
    print("\n--- [模块A, 步骤2: 筛选并提取关键点] ---")
    num_detected = len(result.boxes)
    print(f"初步检测到 {num_detected} 个实例。")

    if num_detected > TARGET_VERTEBRAE_COUNT:
        print(f"检测数量超过目标 {TARGET_VERTEBRAE_COUNT}，将按置信度筛选...")
        conf_scores = result.boxes.conf
        sorted_indices = torch.argsort(conf_scores, descending=True)
        top_indices = sorted_indices[:TARGET_VERTEBRAE_COUNT]

        # 筛选关键点和检测框
        filtered_keypoints_tensor = result.keypoints[top_indices]
        print(f"已筛选出置信度最高的 {len(filtered_keypoints_tensor)} 个实例。")
    else:
        filtered_keypoints_tensor = result.keypoints

    all_keypoints_list = []
    keypoints_xy_tensors = filtered_keypoints_tensor.xy.cpu()

    for instance_kpts_tensor in keypoints_xy_tensors:
        all_keypoints_list.append(instance_kpts_tensor.numpy())

    if not all_keypoints_list:
        print("!! 未能提取任何关键点组。")
        return None, original_image

    all_keypoints_np = np.vstack(all_keypoints_list)

    print(f"成功提取了 {all_keypoints_np.shape[0]} 个关键点。")
    return all_keypoints_np, original_image

if __name__ == "__main__":
    # 【请修改这里】指定您训练好的模型权重的路径
    MODEL_PATH = 'runs/pose/Vertebrae_Pose/weights/best.pt'

    # 【请修改这里】指定您想要进行预测的图片路径
    IMAGE_PATH = r'./data/test/sunhl-1th-01-Mar-2017-311 D AP.jpg'
    extract_keypoints_from_image(MODEL_PATH,IMAGE_PATH)