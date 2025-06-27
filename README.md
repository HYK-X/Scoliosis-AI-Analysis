-----

# 智能脊柱侧弯AI辅助分析系统 (Intelligent Scoliosis AI Analysis System)

[](https://www.python.org/downloads/)
[](https://opensource.org/licenses/MIT)

这是一个基于深度学习的AI辅助诊断平台，旨在实现对脊柱X光图像的自动化分析。系统通过YOLOv11-Pose模型精确识别椎体关键点，运用多种算法自动计算Cobb角，并可调用大语言模型（LLM）生成专业的诊断报告。所有功能均集成在一个交互式的Gradio Web界面中，方便临床和科研人员使用。

An AI-powered diagnostic platform for the automated analysis of scoliosis from X-ray images. This system utilizes the YOLOv11-Pose model for precise vertebra keypoint detection, calculates Cobb angles with multiple algorithms, and generates professional diagnostic reports via a Large Language Model (LLM). All features are integrated into an interactive Gradio web interface for clinical and research use.

*(建议您将此处的图片链接替换为您自己应用的截图)*

## 核心功能 (Key Features)

  - **🚀 高精度关键点检测 (High-Precision Keypoint Detection)**: 使用在自定义数据集上微调的YOLOv11-Pose模型，实现对17个椎骨、共68个关键点的精确识别。
  - **📐 多算法Cobb角计算 (Multi-Algorithm Cobb Angle Calculation)**: 内置多种Cobb角测量方法（如传统的`端板法`和基于曲线拟合的`中心线法`），提供多维度参考。
  - **🤖 AI智能报告生成 (AI-Powered Report Generation)**: 一键调用大语言模型（如DeepSeek API），根据Cobb角测量结果和患者临床信息，自动生成结构化、专业化的诊断分析报告。
  - **🌐 交互式Web界面 (Interactive Web Interface)**: 基于Gradio构建，界面友好直观。用户只需上传图片、调整参数，即可完成完整的分析流程，并实时查看标注图像和诊断报告。
  - **🧩 模块化架构 (Modular Architecture)**: 项目代码结构清晰，分为数据处理、模型训练、关键点检测、角度计算、报告生成和Web界面六大模块，易于维护和二次开发。

## 系统工作流 (Workflow)

```
                       +-------------------------+
                       |   用户上传X光图像       |
                       | (Upload X-Ray Image)    |
                       +-----------+-------------+
                                   |
                                   v
+------------------------+     +-------------------------+     +------------------------+
|   yolo_detector.py     | --> |    Cobb_API_*.py        | --> |      Report_API.py     |
| (关键点检测/Keypoint Det) |     | (Cobb角计算/Angle Calc)  |     | (AI报告生成/AI Report)  |
+------------------------+     +-------------------------+     +------------------------+
            |                            |                               |
            +----------------------------+-------------------------------+
                                   |
                                   v
                       +-----------+-------------+
                       |       web_ui.py         |
                       | (Gradio Web Interface)  |
                       +-------------------------+
                                   |
                                   v
                  +----------------------------------+
                  |  输出：标注图像、Cobb角、诊断报告 |
                  | (Output: Annotated Img, Angles, Report) |
                  +----------------------------------+
```

## 快速开始 (Getting Started)

### 1\. 环境配置 (Prerequisites)

  - Python 3.9 或更高版本
  - PyTorch (建议使用CUDA版本以获得GPU加速)
  - Git

### 2\. 安装 (Installation)

1.  **克隆代码库 (Clone the repository):**

    ```bash
    git clone https://github.com/your-username/Scoliosis-AI-Analysis.git
    cd Scoliosis-AI-Analysis
    ```

2.  **安装依赖 (Install dependencies):**
    建议创建一个虚拟环境。

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

    `requirements.txt` 文件内容:

    ```txt
    ultralytics
    gradio
    torch
    torchvision
    numpy
    opencv-python-headless
    pillow
    seaborn
    matplotlib
    scipy
    pyyaml
    openai
    ```

3.  **配置API密钥 (Configure API Key):**
    AI报告生成功能依赖于DeepSeek API (兼容OpenAI)。请在`Report_API.py`文件中修改您的API密钥：

    ```python
    # 文件: Report_API.py
    API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" # <--- 替换为你的密钥
    ```

4.  **准备模型文件 (Prepare Model Files):**

      - 将您训练好的YOLOv11-Pose模型权重命名为 `best.pt`。
      - 将其放置在路径 `runs/pose/Vertebrae_Pose/weights/` 下。如果目录不存在，请创建它们。
      - 最终模型路径应为：`runs/pose/Vertebrae_Pose/weights/best.pt`。

### 3\. 运行系统 (Usage)

#### 步骤 1: 数据准备与模型训练 (可选)

如果您希望使用自己的数据集进行训练，请遵循以下步骤：

1.  **准备数据**: 参考 `data_processor.py` 的结构，准备您的图像和`.mat`标签文件。
2.  **生成YOLO格式数据集**: 运行 `python data_processor.py`，它将自动处理数据、划分数据集并生成 `Vertebrae_YOLO_Dataset.yaml` 配置文件。
3.  **开始训练**: 运行 `python train_model.py` 来训练您自己的模型。训练出的最佳模型将保存在 `runs/pose/Vertebrae_Pose_Exp1/weights/best.pt`。

#### 步骤 2: 启动Web应用

直接运行主程序来启动Gradio Web界面：

```bash
python web_ui.py
```

程序启动后，会输出一个本地URL (如 `http://127.0.0.1:7860`) 和一个公网URL。在浏览器中打开任一链接即可开始使用。

## 代码模块概览 (Code Modules)

| 文件 (File) | 描述 (Description) |
| :--- | :--- |
| `web_ui.py` | **程序主入口**。构建并启动Gradio交互式Web应用。 |
| `yolo_detector.py` | **关键点检测模块**。加载YOLO模型，对输入图像进行推理，并提取椎体关键点坐标。|
| `Cobb_API_Fix.py` | **增强版Cobb角计算模块**。提供基于端板和中心线的多种精确计算方法，并生成动态边界的可视化结果图。|
| `Cobb_API_Raw.py` | 原始版Cobb角计算模块，提供基础的计算和可视化功能。|
| `Report_API.py` | **AI报告生成模块**。调用大语言模型API，根据分析数据生成结构化的诊断报告。|
| `data_processor.py`| **数据集预处理工具**。将原始标注数据转换为YOLO姿态估计格式，并自动划分训练/验证/测试集。|
| `train_model.py` | **模型训练脚本**。使用`ultralytics`库加载预训练模型，并在自定义数据集上进行微调。|

## 授权许可 (License)

该项目采用 [MIT License] 授权。