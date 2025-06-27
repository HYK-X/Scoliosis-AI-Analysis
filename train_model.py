# -*- coding: utf-8 -*-

from ultralytics import YOLO
import torch


def train_vertebrae_pose_model():
    """
    一个完整的函数，用于加载预训练的YOLOv11姿态模型，
    并使用我们自定义的椎骨数据集进行训练。
    """

    # --- 1. 选择并加载模型 ---

    # 我们选择 'yolov11n-pose.pt'。这是一个在大型数据集(COCO)上预训练过的模型。
    # - 'n' 代表 nano，是YOLOv11系列中最小、最快的模型，非常适合作为起点。
    # - '-pose' 后缀表示这是一个专门用于姿态估计的模型。
    # 加载预训练模型可以极大地加速训练过程并提高最终性能，这被称为“迁移学习”。
    print("--- [步骤 1: 加载预训练模型] ---")
    model = YOLO("yolo11n-pose.yaml").load("yolo11n-pose.pt")  # build from YAML and transfer weights
    print("模型 'yolov11n--pose.pt' 加载成功。")
    # --- 2. 设置数据集配置文件 ---
    # 指定我们之前创建的数据集配置文件的路径。
    # YOLO框架将从这个文件中读取所有关于数据集的信息：
    # - 训练/验证/测试集的图片路径
    # - 类别名称 ('vertebra')
    # - 关键点形状和翻转索引
    data_yaml_path = 'Vertebrae_YOLO_Dataset.yaml'
    print(f"--- [步骤 2: 指定数据集配置文件] ---")
    print(f"配置文件路径: {data_yaml_path}")
    # --- 3. 开始模型训练 ---
    print("\n--- [步骤 3: 开始模型训练] ---")
    print("训练即将开始，您可以在终端中看到实时进度...")
    print("训练完成后，所有结果将保存在 'runs/pose/train' 目录下。")
    # 调用 model.train() 方法来启动训练过程。
    # 我们在这里设置一些关键的训练参数：
    results = model.train(
        data=data_yaml_path,  # 数据集配置文件
        epochs=150,  # 训练轮次：整个数据集被完整训练150次。可以根据效果调整，初期可以设为50-100。
        # imgsz=640,  # 输入图像尺寸：将所有图片统一缩放到 640x640 像素进行训练。
        batch=8,  # 批处理大小：每次迭代喂给模型8张图片。如果您的显存(GPU VRAM)较大，可以适当调高此值以加快训练速度。
        device=0,  # 指定使用的计算设备。0 通常代表第一块GPU。如果您没有GPU或只想用CPU，可以设为 'cpu'。
        name='Vertebrae_Pose_Exp1'  # 训练任务的名称：所有结果（权重、日志、图表）将被保存在 'runs/pose/Vertebrae_Pose_Exp1' 文件夹中。
    )
    # --- 4. 训练完成 ---
    print("\n🎉🎉🎉 [训练完成!] 🎉🎉🎉")
    print(f"所有训练结果、权重和日志都已保存在目录: {results.save_dir}")
    print("最重要的文件是 'best.pt'，这是您训练出的性能最好的模型权重！")


if __name__ == '__main__':
    # 检查是否有可用的GPU
    print("正在检查设备...")
    if torch.cuda.is_available():
        print(f"发现可用的 CUDA 设备 (GPU): {torch.cuda.get_device_name(0)}")
    else:
        print("未发现可用的 CUDA 设备，将使用 CPU 进行训练。注意：使用CPU训练会非常慢。")

    # 运行主训练函数
    train_vertebrae_pose_model()
