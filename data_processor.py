# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import scipy.io
import shutil
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import train_test_split
import random

# --- 1. 专业级用户配置区 ---

# 输入/输出文件夹名称
BASE_DIR = '.'
DATA_FOLDER = 'data'
LABELS_FOLDER = 'labels'
OUTPUT_FOLDER = 'Vertebrae_YOLO_Dataset'  # 使用更有意义的输出文件夹名

# .mat 文件中的关键点变量名
MAT_KEYPOINT_VAR_NAME = 'p2'

# --- 关键点与类别配置 (针对“椎骨”场景) ---

# 1. 类别定义
YOLO_CLASS_ID = 0
CLASS_NAME = 'vertebra'  # 类别名：椎骨

# 2. 关键点结构定义
NUM_KEYPOINTS_PER_GROUP = 4  # 每组4个点代表一节椎骨
KEYPOINT_DIMS = 2  # 2D关键点 (x, y)

# 3. 骨架连接定义 (SKELETON)
#    定义一节椎骨内部4个点的连接关系，用于可视化。
#    假设您的4个点是按顺时针或逆时针排列的四边形角点。
#    索引从0开始，[0, 1] 表示连接第1个点和第2个点。
SKELETON = [[0, 1], [1, 2], [2, 3], [3, 0]]  # <--- 请根据您4个点的实际连接方式修改

# 4. 水平翻转索引 (FLIP_IDX)
#    这对于数据增强至关重要！请仔细定义。
#    例子：假设点0/3是左侧点，点1/2是右侧点。
#    翻转后，0和1互换，3和2互换。则FLIP_IDX应为 [1, 0, 3, 2]。
#    如果您的4个点没有左右对称性，或者排列方式不规则，请保持 [0, 1, 2, 3]。
FLIP_IDX = [1, 0, 3, 2]  # <--- 这是一个示例，请务必根据您的点位定义修改！

# 5. 数据集划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1  # 三者相加必须为1.0

# --- 全局变量 ---
last_image_info_for_viz = {"image_path": None, "all_keypoints": None}


def split_data():
    """收集所有数据，并按比例随机划分为训练、验证、测试集。"""
    print("--- [步骤 1: 划分数据集] ---")
    all_image_names = []
    # 从原始的training和test文件夹中收集所有图片名
    for split_folder in ['training', 'test']:
        folder_path = os.path.join(BASE_DIR, DATA_FOLDER, split_folder)
        if os.path.exists(folder_path):
            all_image_names.extend(
                [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # 去重并随机打乱
    all_image_names = sorted(list(set(all_image_names)))
    random.shuffle(all_image_names)

    total_count = len(all_image_names)
    if total_count == 0:
        print("!! 错误: 在data/training和data/test中未找到任何图片。")
        return None, None, None

    # 使用sklearn进行划分
    train_val_names, test_names = train_test_split(all_image_names, test_size=TEST_RATIO, random_state=42)
    # 在剩余部分中再划分训练集和验证集
    val_size_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_names, val_names = train_test_split(train_val_names, test_size=val_size_adjusted, random_state=42)

    print(f"所有图片总数: {total_count}")
    print(f"训练集数量: {len(train_names)}")
    print(f"验证集数量: {len(val_names)}")
    print(f"测试集数量: {len(test_names)}")

    return train_names, val_names, test_names


def process_split(image_names, split_name):
    """根据给定的图片列表和划分名称(train/val/test)来处理数据。"""
    global last_image_info_for_viz
    print(f"\n========== 开始处理 '{split_name}' 数据集 ==========")

    output_images_path = os.path.join(BASE_DIR, OUTPUT_FOLDER, 'images', split_name)
    output_labels_path = os.path.join(BASE_DIR, OUTPUT_FOLDER, 'labels', split_name)
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)

    for filename in image_names:
        # 在原始的training和test文件夹中寻找源文件
        original_img_path = None
        for original_split in ['training', 'test']:
            path = os.path.join(BASE_DIR, DATA_FOLDER, original_split, filename)
            if os.path.exists(path):
                original_img_path = path
                break

        if not original_img_path:
            print(f"!! 警告: 找不到图片源文件 {filename}，跳过。")
            continue

        label_path = os.path.join(BASE_DIR, LABELS_FOLDER, os.path.basename(os.path.dirname(original_img_path)),
                                  filename + '.mat')
        if not os.path.exists(label_path):
            print(f"!! 警告: 找不到 {filename} 的标签文件，跳过。")
            continue

        # ... (后续的文件处理逻辑与之前类似) ...
        try:
            img = cv2.imread(original_img_path)
            if img is None: raise IOError("无法读取图片")
            h, w, _ = img.shape
            mat = scipy.io.loadmat(label_path)
            all_kpts = mat[MAT_KEYPOINT_VAR_NAME].astype(np.float32)
        except Exception as e:
            print(f"!! 错误: 读取文件 {filename} 时出错: {e}，跳过。")
            continue

        yolo_lines = []
        for i in range(0, len(all_kpts), NUM_KEYPOINTS_PER_GROUP):
            group = all_kpts[i: i + NUM_KEYPOINTS_PER_GROUP]
            if len(group) != NUM_KEYPOINTS_PER_GROUP: continue

            x_coords, y_coords = group[:, 0], group[:, 1]
            x_min, y_min, x_max, y_max = np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)

            cx, cy = (x_min + x_max) / 2 / w, (y_min + y_max) / 2 / h
            bw, bh = (x_max - x_min) / w, (y_max - y_min) / h

            norm_kpts = group.copy()
            norm_kpts[:, 0] /= w
            norm_kpts[:, 1] /= h

            bbox_str = f"{YOLO_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            kpts_str = " ".join([f"{c:.6f}" for c in norm_kpts.flatten()])
            yolo_lines.append(f"{bbox_str} {kpts_str}")

        if yolo_lines:
            txt_path = os.path.join(output_labels_path, os.path.splitext(filename)[0] + '.txt')
            with open(txt_path, 'w') as f:
                f.write("\n".join(yolo_lines))
            shutil.copy(original_img_path, os.path.join(output_images_path, filename))

            last_image_info_for_viz["image_path"] = os.path.join(output_images_path, filename)
            last_image_info_for_viz["all_keypoints"] = all_kpts

    print(f"========== '{split_name}' 数据集处理完毕 ==========")


def generate_yaml_file():
    """生成最终的 dataset.yaml 文件。"""
    print("\n--- [步骤 3: 生成 YAML 配置文件] ---")
    yaml_data = {
        'path': os.path.abspath(OUTPUT_FOLDER),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'kpt_shape': [NUM_KEYPOINTS_PER_GROUP, KEYPOINT_DIMS],
        'flip_idx': FLIP_IDX,
        'names': {YOLO_CLASS_ID: CLASS_NAME},
        'skeleton': SKELETON  # 添加骨架信息，便于其他工具使用
    }
    yaml_file_path = os.path.join(BASE_DIR, f'{OUTPUT_FOLDER}.yaml')
    try:
        with open(yaml_file_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"✅ [成功] 已在 '{yaml_file_path}' 创建配置文件。")
    except Exception as e:
        print(f"❌ [失败] 创建YAML文件时出错: {e}")


def visualize_last_image():
    """可视化最后一张图片的标注结果，包含骨架连接。"""
    print("\n--- [步骤 4: 可视化样本检查] ---")
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False
    except:
        print("!! 警告: 设置中文字体失败。")

    info = last_image_info_for_viz
    if not info["image_path"]: print("!! 没有可供可视化的图片。"); return

    image = cv2.imread(info["image_path"])
    all_keypoints = info["all_keypoints"]
    bbox_colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    for i in range(0, len(all_keypoints), NUM_KEYPOINTS_PER_GROUP):
        group = all_keypoints[i: i + NUM_KEYPOINTS_PER_GROUP]
        if len(group) != NUM_KEYPOINTS_PER_GROUP: continue

        # 绘制BBox
        x_min, y_min, x_max, y_max = np.min(group[:, 0]), np.min(group[:, 1]), np.max(group[:, 0]), np.max(group[:, 1])
        color = bbox_colors[(i // NUM_KEYPOINTS_PER_GROUP) % len(bbox_colors)]
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

        # 绘制骨架连接线
        for p1_idx, p2_idx in SKELETON:
            p1 = tuple(group[p1_idx].astype(int))
            p2 = tuple(group[p2_idx].astype(int))
            cv2.line(image, p1, p2, color, 2)

    for idx, (x, y) in enumerate(all_keypoints):
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(image, str(idx % NUM_KEYPOINTS_PER_GROUP), (int(x) + 7, int(y) + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 10));
    plt.imshow(image_rgb)
    plt.title("可视化样本预览：椎骨(Vertebra)标注", fontsize=16)
    plt.axis('off');
    plt.show()


def main():
    """主执行函数"""
    # 1. 划分数据集
    train_files, val_files, test_files = split_data()
    if train_files is None: return

    # 2. 分别处理三个数据集
    print("\n--- [步骤 2: 转换数据格式] ---")
    process_split(train_files, 'train')
    process_split(val_files, 'val')
    process_split(test_files, 'test')

    # 3. 生成YAML文件
    generate_yaml_file()

    # 4. 可视化检查
    visualize_last_image()

    print("\n🎉🎉🎉 [祝贺您] 专业级数据集制作流程全部完成！🎉🎉🎉")


if __name__ == '__main__':
    main()