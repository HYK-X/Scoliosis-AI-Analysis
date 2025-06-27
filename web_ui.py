# -*- coding: utf-8 -*-

"""
智能脊柱侧弯分析系统 - Gradio Web应用 (修复版)
===============================================
版本: v2.0.3
更新: 修复Gradio组件参数兼容性问题
"""

import os
import cv2
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

# 导入核心模块（如缺失则使用模拟函数）
try:
    from yolo_detector import extract_keypoints_from_image
    from Cobb_API_Raw import process_image_with_points
    from Cobb_API_Fix import process_image_with_enhanced_calculation
    from Report_API import stream_ai_report

    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 模块导入警告: {e}")
    print("🔧 使用模拟模式运行")
    MODULES_AVAILABLE = False


    # ==================== 模拟函数区域 ====================
    def extract_keypoints_from_image(model_path, image_path, conf_threshold):
        """模拟关键点提取"""
        print("🔍 执行关键点提取...")
        time.sleep(1.5)

        image = cv2.imread(image_path)
        if image is None:
            return None, None

        h, w = image.shape[:2]
        keypoints = []

        # 生成模拟关键点
        for i in range(1, 18):
            for j in range(4):
                x = w * (0.45 + np.random.rand() * 0.1)
                y = h * (0.1 + i * 0.045 + np.random.rand() * 0.01)
                keypoints.append([x, y, i, j])

        return np.array(keypoints), image


    def process_image_with_points(image, keypoints):
        """模拟标准Cobb角计算"""
        print("📐 执行标准Cobb角计算...")
        time.sleep(1.2)

        cv2.putText(image, "Upper Thoracic: 25.3°", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
        cv2.putText(image, "Main Thoracic: 18.7°", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
        cv2.putText(image, "Thoracolumbar: 12.1°", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

        angles = {
            "上胸弯 (PT)": "25.3°",
            "主胸弯 (MT)": "18.7°",
            "胸腰弯/腰弯 (TL/L)": "12.1°"
        }
        return image, angles


    def process_image_with_enhanced_calculation(image, keypoints, method):
        """模拟增强Cobb角计算"""
        print(f"🔬 执行{method}方法计算...")
        time.sleep(1.3)

        color = (0, 200, 0) if method == 'centerline' else (255, 100, 0)
        method_name = "中心线法" if method == 'centerline' else "终板法"

        cv2.putText(image, f"PT ({method_name}): 26.1°", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(image, f"MT ({method_name}): 19.2°", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(image, f"TL ({method_name}): 11.8°", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        angles = {
            "上胸弯 (PT)": "26.1°",
            "主胸弯 (MT)": "19.2°",
            "胸腰弯/腰弯 (TL/L)": "11.8°"
        }
        return image, angles


    def stream_ai_report(patient_info):
        """模拟AI报告生成"""
        print("🤖 生成AI诊断报告...")

        cobb_angles = patient_info.get('cobb_angles', {})
        patient_age = patient_info.get('age', '无')
        patient_sex = patient_info.get('sex', '无')
        symptoms = patient_info.get('symptoms', '无')
        skeletal_maturity = patient_info.get('skeletal_maturity', '无')

        # 计算最大角度确定严重程度
        max_angle_val = 0
        if cobb_angles:
            try:
                max_angle_val = max(float(v.replace("°", "")) for v in cobb_angles.values())
            except (ValueError, TypeError):
                max_angle_val = 20

        if max_angle_val < 10:
            severity = '生理性'
        elif max_angle_val < 25:
            severity = '轻度'
        elif max_angle_val < 45:
            severity = '中度'
        else:
            severity = '重度'

        # 生成诊断报告
        report_content = f"""### 🔬 **影像学分析报告**

**患者基本信息:**
- 年龄: **{patient_age}** {'岁' if patient_age != '无' else ''}
- 性别: **{patient_sex}**
- 骨骼成熟度: **{skeletal_maturity}**
- 主要症状: {symptoms}

**Cobb角测量结果:**"""

        for angle_name, angle_value in cobb_angles.items():
            report_content += f"\n- **{angle_name}**: <span style='color: #1E88E5; font-weight: bold;'>{angle_value}</span>"

        report_content += f"""

**影像学表现:**
脊柱正位X光片显示脊柱侧弯畸形，主要弯曲位于胸段，最大Cobb角约为{max_angle_val:.1f}度。椎体形态基本正常，未见明显先天性异常。

### 🩺 **临床诊断**

**主要诊断:** 青少年特发性脊柱侧弯 (AIS)

**分型评估:**
- **Lenke分型**: 初步判断为Lenke 1型（主胸弯型）
- **严重程度**: **{severity}**脊柱侧弯
- **进展风险**: {'高风险' if patient_age != '无' and patient_age and int(patient_age) < 16 else '中等风险'}

### 💊 **治疗建议**

**保守治疗方案:**"""

        if max_angle_val < 25:
            report_content += """
1. **观察随访**: 每6个月复查X光片
2. **功能锻炼**: Schroth三维矫形体操
3. **姿势管理**: 避免不良姿势，加强核心训练
4. **生活指导**: 双肩背包，避免单侧负重"""
        elif max_angle_val < 45:
            report_content += """
1. **支具治疗**: 建议佩戴TLSO矫形器
2. **定期随访**: 每3-4个月复查
3. **物理治疗**: Schroth方法训练
4. **心理支持**: 关注心理健康"""
        else:
            report_content += """
1. **手术评估**: 建议脊柱外科会诊
2. **术前准备**: 完善相关检查
3. **手术方式**: 考虑后路内固定融合术
4. **康复计划**: 制定术后康复方案"""

        report_content += f"""

### ⚠️ **注意事项**

1. **随访频率**: 建议每{3 if max_angle_val > 25 else 6}个月复查
2. **运动限制**: {'避免高强度运动' if max_angle_val > 30 else '可适度运动'}
3. **家长须知**: 密切观察姿态变化

**免责声明:** 本报告由AI生成，仅供参考。最终诊疗需专业医师确认。

---
*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*
*系统版本: v2.0.3*"""

        # 流式输出
        for i, char in enumerate(report_content):
            yield char
            if char in ['*', '#', ':', '：']:
                time.sleep(0.015)
            else:
                time.sleep(0.003)

# ==================== 系统配置 ====================
MODEL_PATH = 'runs/pose/Vertebrae_Pose/weights/best.pt'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 工具函数 ====================
def optimize_image_display(image, target_width=800, target_height=600):
    """优化图像显示尺寸"""
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]

        width_ratio = target_width / width
        height_ratio = target_height / height
        scale_ratio = min(width_ratio, height_ratio)

        if scale_ratio < 1.0:
            new_width = int(width * scale_ratio)
            new_height = int(height * scale_ratio)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return image


def validate_patient_info(age, sex, symptoms):
    """验证患者信息完整性"""
    missing_fields = []

    # 修改验证逻辑，将"无"视为有效输入
    if not age or age <= 0:
        missing_fields.append("年龄")
    if not sex or sex == "无":
        missing_fields.append("性别")
    if not symptoms or len(symptoms.strip()) < 2 or symptoms.strip() == "无":
        missing_fields.append("症状描述")

    return len(missing_fields) == 0, missing_fields


# ==================== 核心处理逻辑 ====================
def execute_spine_analysis_pipeline(input_image, calculation_method, patient_age, patient_sex,
                                    patient_symptoms, skeletal_maturity, other_findings, conf_threshold):
    """脊柱分析主处理流程"""

    # 输入验证
    if input_image is None:
        error_msg = "❌ 请先上传脊柱X光图像"
        yield None, error_msg, "### 📤 等待图像上传\n请上传清晰的脊柱正位X光图像", ""
        return

    display_image_rgb = None
    angles_display_text = "⏳ 等待分析结果..."
    report_part1 = "### 🔄 系统准备中...\n正在初始化AI分析引擎..."
    report_part2 = ""

    try:
        # 阶段1: 关键点提取
        print(f"🚀 开始分析 | 方法: {calculation_method}")
        yield None, "🔍 第1步：脊椎关键点检测中...", "### 🔄 分析进行中\n**当前阶段:** 关键点检测", ""

        opencv_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
        temp_image_path = "temp_spine_image.jpg"
        cv2.imwrite(temp_image_path, opencv_image)

        detected_keypoints, processed_image = extract_keypoints_from_image(
            MODEL_PATH, temp_image_path, conf_threshold=conf_threshold
        )

        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        if detected_keypoints is None or len(detected_keypoints) == 0:
            error_msg = f"⚠️ 关键点检测失败\n• 调低置信度阈值 (当前: {conf_threshold})\n• 确保图像清晰"
            yield None, error_msg, f"### ❌ 检测失败\n{error_msg}", ""
            return

        print(f"✅ 检测到 {len(detected_keypoints)} 个关键点")

        # 阶段2: Cobb角计算
        yield None, f"📐 第2步：{calculation_method}计算Cobb角...", "### 🔄 分析进行中\n**当前阶段:** Cobb角计算", ""

        if calculation_method == "标准方法":
            annotated_image, measured_angles = process_image_with_points(processed_image, detected_keypoints)
        elif calculation_method == "中心线方法":
            annotated_image, measured_angles = process_image_with_enhanced_calculation(
                processed_image, detected_keypoints, method='centerline'
            )
        elif calculation_method == "终板方法":
            annotated_image, measured_angles = process_image_with_enhanced_calculation(
                processed_image, detected_keypoints, method='endplate'
            )
        else:
            yield None, f"❌ 未知计算方法: {calculation_method}", "### ❌ 参数错误", ""
            return

        # 阶段3: 角度数据处理
        processed_angles = {}
        if measured_angles:
            angle_mapping = {
                'PT': '上胸弯 (PT)',
                'MT': '主胸弯 (MT)',
                'TL': '胸腰弯/腰弯 (TL/L)',
                'L': '腰弯 (L)'
            }

            normalized_angles = {angle_mapping.get(k, k): v for k, v in measured_angles.items()}

            for angle_name, angle_value in normalized_angles.items():
                try:
                    numeric_value = abs(float(str(angle_value).replace('°', '')))

                    if numeric_value > 90.0:
                        numeric_value = 180.0 - numeric_value

                    if numeric_value > 0.5:
                        processed_angles[angle_name] = f"{numeric_value:.1f}°"

                except (ValueError, TypeError) as e:
                    print(f"⚠️ 角度处理错误: {angle_name} = {angle_value}")
                    continue

        # 阶段4: 图像显示优化
        optimized_image = optimize_image_display(annotated_image)
        display_image_rgb = cv2.cvtColor(optimized_image, cv2.COLOR_BGR2RGB)

        # 格式化角度显示
        if processed_angles:
            angles_display_text = "### 📊 **Cobb角度测量结果**\n\n"
            for angle_name, angle_value in processed_angles.items():
                numeric_val = float(angle_value.replace('°', ''))
                if numeric_val < 10:
                    color_style = "color: #4CAF50; font-weight: bold;"
                elif numeric_val < 25:
                    color_style = "color: #FF9800; font-weight: bold;"
                else:
                    color_style = "color: #F44336; font-weight: bold;"

                angles_display_text += f"- **{angle_name}**: <span style='{color_style}'>{angle_value}</span>\n"

            angles_display_text += f"\n*计算方法: {calculation_method}*"
        else:
            angles_display_text = "⚠️ **未检测到有效Cobb角度**\n\n可能原因:\n- 图像质量不佳\n- 脊柱无明显侧弯"

        # 阶段5: AI报告生成
        # 修改验证逻辑，允许默认值"无"
        has_basic_info = (patient_age and patient_age > 0) or (patient_sex and patient_sex != "无") or (
                    patient_symptoms and patient_symptoms.strip() != "无" and len(patient_symptoms.strip()) > 2)

        if processed_angles :
            print("🤖 生成AI诊断报告...")
            yield display_image_rgb, angles_display_text, "### 🤖 正在生成AI诊断报告...", ""

            patient_data = {
                "age": patient_age if patient_age and patient_age > 0 else "无",
                "sex": patient_sex if patient_sex and patient_sex != "无" else "无",
                "symptoms": patient_symptoms if patient_symptoms and patient_symptoms.strip() != "无" else "无",
                "cobb_angles": processed_angles,
                "skeletal_maturity": skeletal_maturity if skeletal_maturity and skeletal_maturity.strip() != "无" else "无",
                "other_findings": other_findings if other_findings and other_findings.strip() != "无" else "无"
            }

            try:
                report_generator = stream_ai_report(patient_data)
                accumulated_report = ""

                for report_chunk in report_generator:
                    accumulated_report += report_chunk

                    total_length = len(accumulated_report)
                    split_position = int(total_length * 0.5)

                    section_break = accumulated_report.rfind("### ", 0, split_position)
                    if section_break != -1:
                        split_position = section_break

                    report_part1 = accumulated_report[:split_position]
                    report_part2 = accumulated_report[split_position:]

                    yield display_image_rgb, angles_display_text, report_part1, report_part2
                    time.sleep(0.01)

            except Exception as report_error:
                error_report = f"""### ❌ **AI报告生成异常**

**错误信息:** `{str(report_error)}`

**手动诊断参考:**
- < 10°: 生理性变化
- 10-25°: 轻度侧弯
- 25-45°: 中度侧弯  
- > 45°: 重度侧弯"""
                yield display_image_rgb, angles_display_text, error_report, ""

        elif not processed_angles:
            report_part1 = """### ⚠️ **无法生成诊断报告**

**原因:** 未检测到有效的Cobb角度

**建议:**
1. 确保图像清晰，对比度良好
2. 使用标准脊柱正位摄影
3. 尝试降低置信度阈值
4. 重新上传高质量图像"""

        else:
            report_part1 = f"""### 📝 **信息不足，生成基础报告**

**测量结果有效，但患者信息较少**

为获得更详细的AI诊断报告，建议：
1. 展开"患者临床信息"面板
2. 填写年龄、性别等基本信息
3. 描述主要症状表现
4. 重新开始分析

**当前测量结果仍然有效，可供临床参考。**"""

        yield display_image_rgb, angles_display_text, report_part1, report_part2

    except Exception as system_error:
        error_details = f"""### 🚨 **系统处理异常**

**错误:** `{str(system_error)}`

**解决方案:**
1. 重新上传图像
2. 调整参数设置
3. 刷新页面重试"""

        print(f"❌ 系统异常: {system_error}")
        yield None, f"❌ 处理失败: {str(system_error)}", error_details, ""


# ==================== 示例报告功能 ====================
def generate_demo_report_initial():
    """示例报告初始文本"""
    return """### 📄 **AI诊断报告示例**

点击 **"生成示例诊断报告"** 查看完整AI报告样例。

**示例内容:**
- 详细影像学分析
- 专业临床诊断
- 个性化治疗建议
- 随访指导方案"""


def generate_demo_report_stream():
    """流式生成示例报告"""
    demo_patient_data = {
        "age": 15,
        "sex": "女",
        "symptoms": "体检发现姿态异常，右肩高于左肩",
        "cobb_angles": {
            "上胸弯 (PT)": "23.5°",
            "主胸弯 (MT)": "31.8°",
            "胸腰弯/腰弯 (TL/L)": "18.2°"
        },
        "skeletal_maturity": "Risser 2级",
        "other_findings": "椎体形态基本正常"
    }

    try:
        accumulated_content = ""
        report_stream = stream_ai_report(demo_patient_data)

        for content_chunk in report_stream:
            accumulated_content += content_chunk
            yield accumulated_content
            time.sleep(0.008)

    except Exception as demo_error:
        yield f"""### ❌ **示例报告生成失败**

**错误:** `{str(demo_error)}`

**实际功能包括:**
- 智能影像学分析
- 个性化诊断建议
- 专业治疗方案
- 详细随访指导"""


# ==================== Gradio界面构建 ====================
def build_optimized_gradio_interface():
    """构建优化的Gradio界面"""

    # CSS样式定义
    medical_css = """
    :root {
        --primary-blue: #1976D2; --primary-blue-dark: #0D47A1;
        --success-green: #388E3C; --warning-orange: #F57C00;
        --error-red: #D32F2F; --bg-primary: #FAFAFA;
        --bg-secondary: #FFFFFF; --text-primary: #212121;
        --border-light: #E0E0E0; --shadow-light: 0 2px 8px rgba(0,0,0,0.08);
        --shadow-medium: 0 4px 16px rgba(0,0,0,0.12);
        --transition-medium: all 0.3s ease; --radius-medium: 12px;
    }

    body {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, 'Microsoft YaHei', sans-serif;
    }

    .gradio-container {
        max-width: 1600px !important;
        margin: 0 auto !important;
        padding: 20px !important;
    }

    .hero-header {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-blue-dark) 100%);
        border-radius: var(--radius-medium);
        padding: 40px 30px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: var(--shadow-medium);
    }

    .hero-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0 0 15px 0;
    }

    .hero-header p {
        font-size: 1.3rem;
        opacity: 0.95;
        margin: 0;
    }

    .medical-card {
        background: var(--bg-secondary);
        border-radius: var(--radius-medium);
        box-shadow: var(--shadow-light);
        padding: 25px;
        margin-bottom: 20px;
        border: 1px solid var(--border-light);
        transition: var(--transition-medium);
    }

    .medical-card:hover {
        box-shadow: var(--shadow-medium);
        transform: translateY(-3px);
    }

    .card-header {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 2px solid var(--primary-blue);
    }

    .card-header h2 {
        font-size: 1.6rem;
        font-weight: 600;
        margin: 0;
        color: var(--primary-blue-dark);
    }

    .alert-box {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        border-left: 5px solid var(--warning-orange);
        border-radius: 6px;
        padding: 20px 25px;
        margin: 25px 0;
        box-shadow: var(--shadow-light);
    }

    .alert-box h3 {
        color: #E65100;
        font-weight: 600;
        margin: 0 0 10px 0;
    }

    .btn-primary {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-blue-dark) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-medium) !important;
        padding: 16px 32px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: var(--transition-medium) !important;
        box-shadow: var(--shadow-light) !important;
    }

    .btn-primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-medium) !important;
    }

    .btn-secondary {
        background: linear-gradient(135deg, var(--success-green) 0%, #2E7D32 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-medium) !important;
        padding: 14px 28px !important;
        transition: var(--transition-medium) !important;
    }

    .result-image img {
        border-radius: var(--radius-medium) !important;
        box-shadow: var(--shadow-medium) !important;
        max-width: 100% !important;
    }

    .report-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 25px;
        margin-top: 20px;
    }

    .report-column {
        background: var(--bg-secondary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-medium);
        padding: 20px;
        max-height: 600px;
        overflow-y: auto;
        box-shadow: var(--shadow-light);
    }

    /* 可折叠面板样式优化 */
    .gradio-accordion {
        border-radius: var(--radius-medium) !important;
        border: 1px solid var(--border-light) !important;
        box-shadow: var(--shadow-light) !important;
        margin-bottom: 15px !important;
    }

    .gradio-accordion summary {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%) !important;
        color: var(--primary-blue-dark) !important;
        font-weight: 600 !important;
        padding: 15px 20px !important;
        border-radius: var(--radius-medium) !important;
    }

    .gradio-accordion[open] summary {
        border-bottom: 1px solid var(--border-light) !important;
        border-radius: var(--radius-medium) var(--radius-medium) 0 0 !important;
    }

    @media (max-width: 1200px) {
        .report-container {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 768px) {
        .hero-header h1 { font-size: 2.2rem; }
        .medical-card { padding: 20px; }
    }
    """

    # 构建界面
    with gr.Blocks(css=medical_css, title="智能脊柱侧弯分析系统", theme=gr.themes.Soft()) as interface:
        # 页面标题
        gr.HTML("""
        <div class="hero-header">
            <h1>🏥 智能脊柱侧弯分析系统</h1>
            <p>基于深度学习的AI辅助诊断平台 | 精准测量 · 智能分析 · 专业报告</p>
        </div>
        """)

        # 安全提示
        gr.HTML("""
        <div class="alert-box">
            <h3>⚠️ 医疗安全提示</h3>
            <p><strong>本系统为科研教学辅助工具</strong>，分析结果需由专业医师结合临床情况最终确认，不可作为独立诊断依据。</p>
        </div>
        """)

        # 主功能区域
        with gr.Row():
            # 左侧输入面板
            with gr.Column(scale=1):
                with gr.Column(elem_classes="medical-card"):
                    gr.HTML('<div class="card-header"><h2>📤 图像上传与参数配置</h2></div>')

                    input_image = gr.Image(
                        label="🖼️ 上传脊柱正位X光图像",
                        type="pil",
                        height=300
                    )

                    gr.Markdown("### ⚙️ 分析参数设置")

                    calculation_method = gr.Radio(
                        choices=["标准方法", "中心线方法", "终板方法"],
                        value="标准方法",
                        label="Cobb角计算算法"
                    )

                    conf_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.25,
                        step=0.05,
                        label="关键点检测置信度"
                    )

                    # 将患者临床信息改为可折叠面板，默认关闭
                    with gr.Accordion("👤 患者临床信息 (可选)", open=False):
                        gr.Markdown("*填写患者信息可获得更详细的AI诊断报告*")

                        with gr.Row():
                            # 修复：移除placeholder参数
                            patient_age = gr.Number(
                                label="年龄 (岁) - 留空表示未知",
                                precision=0,
                                minimum=0,
                                maximum=100,
                                value=None
                            )
                            patient_sex = gr.Radio(
                                choices=["男", "女", "无"],
                                value="无",
                                label="性别"
                            )

                        patient_symptoms = gr.Textbox(
                            label="主要症状描述",
                            value="无",
                            lines=3,
                            info="详细描述患者症状或保持'无'"
                        )

                        skeletal_maturity = gr.Textbox(
                            label="骨骼成熟度评估",
                            value="无",
                            info="如：Risser 2级，或保持'无'"
                        )

                        other_findings = gr.Textbox(
                            label="其他影像学发现",
                            value="无",
                            lines=2,
                            info="椎体形态、矢状面曲度等，或保持'无'"
                        )

                    gr.HTML("<br>")
                    analyze_button = gr.Button(
                        "🚀 开始智能分析",
                        elem_classes="btn-primary",
                        size="lg"
                    )

            # 右侧结果面板
            with gr.Column(scale=2):
                with gr.Column(elem_classes="medical-card"):
                    gr.HTML('<div class="card-header"><h2>🔬 分析结果与AI诊断报告</h2></div>')

                    result_image = gr.Image(
                        label="Cobb角标注结果",
                        elem_classes="result-image",
                        height=550,
                        show_label=False
                    )

                    angles_result = gr.Markdown(
                        "### 📊 **Cobb角度测量结果**\n\n⏳ 等待开始分析..."
                    )

                    gr.Markdown("### 🤖 **AI智能诊断报告**")
                    with gr.Row(elem_classes="report-container"):
                        with gr.Column(elem_classes="report-column"):
                            report_column_1 = gr.Markdown(
                                "### 📋 等待生成报告...\n\n上传图像后开始分析。如需详细报告，请展开左侧'患者临床信息'面板填写相关信息。")
                        with gr.Column(elem_classes="report-column"):
                            report_column_2 = gr.Markdown("")

        # 扩展功能
        with gr.Accordion("💡 查看AI报告示例", open=False):
            with gr.Row():
                demo_report_button = gr.Button(
                    "📄 生成示例诊断报告",
                    elem_classes="btn-secondary"
                )

            demo_report_output = gr.Markdown(
                value=generate_demo_report_initial()
            )

        with gr.Accordion("📚 系统使用指南", open=False):
            gr.Markdown("""
            ### 使用流程
            1. **图像上传**: 上传清晰的脊柱正位X光片
            2. **参数设置**: 选择计算方法，调整置信度
            3. **信息填写** (可选): 展开"患者临床信息"面板，填写详细信息可获得更完整的AI诊断报告
            4. **开始分析**: 点击分析按钮获取结果

            ### 最佳实践
            - 使用高质量X光图像
            - 患者信息可选填，但完整信息有助于生成更详细的诊断报告
            - 多方法对比验证
            - 专业医师最终确认

            ### 默认值说明
            - 患者临床信息面板默认折叠，所有字段默认填写"无"
            - 即使不填写患者信息，系统仍可进行Cobb角测量
            - 填写完整患者信息可获得详细的AI诊断报告
            """)

        with gr.Accordion("🔧 技术信息", open=False):
            system_status = "🟢 完整模式" if MODULES_AVAILABLE else "🟡 模拟模式"
            model_status = "🟢 已加载" if os.path.exists(MODEL_PATH) else "🔴 缺失"

            gr.Markdown(f"""
            ### 系统状态
            - 核心模块: {system_status}
            - AI模型: {model_status}
            - 界面版本: v2.0.3

            ### 技术特性
            - YOLO关键点检测
            - 多种Cobb角算法
            - AI智能报告生成
            - 流式实时输出
            - 可折叠患者信息面板
            - 兼容性优化
            """)

        # 事件绑定
        analyze_button.click(
            fn=execute_spine_analysis_pipeline,
            inputs=[
                input_image, calculation_method, patient_age, patient_sex,
                patient_symptoms, skeletal_maturity, other_findings, conf_threshold
            ],
            outputs=[result_image, angles_result, report_column_1, report_column_2]
        )

        demo_report_button.click(
            fn=generate_demo_report_stream,
            inputs=None,
            outputs=demo_report_output
        )

    return interface


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    print("🚀 启动智能脊柱侧弯分析系统 v2.0.3...")

    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ 模型文件未找到: {MODEL_PATH}")
        print("💡 系统将以模拟模式运行")

    if not MODULES_AVAILABLE:
        print("🔧 模拟模式运行，功能完整可测试")
    else:
        print("✅ 完整功能模式运行")

    print("🌐 构建Web界面...")

    app = build_optimized_gradio_interface()
    app.queue()

    print("🎉 启动成功！")
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        inbrowser=True,
        show_error=True
    )
