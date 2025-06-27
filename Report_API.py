# 文件名: Report_API.py
# -*- coding: utf-8 -*-

# 请先安装所需库: pip install openai
from openai import OpenAI
import os
import time

# --- 1. 配置 ---

# 配置DeepSeek API客户端
# 建议从环境变量中读取API密钥，这样更安全。
# 在您的系统中设置环境变量 DEEPSEEK_API_KEY，然后使用 os.getenv("DEEPSEEK_API_KEY")
# 为了方便演示，这里直接写入。请替换为您自己的密钥。
API_KEY = ""
BASE_URL = ""

try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
except Exception as e:
    print(f"初始化OpenAI客户端时出错: {e}")
    client = None


# --- 2. 核心功能函数（流式版本）---

def stream_ai_report(patient_data: dict):
    """
    根据患者数据字典，流式调用AI生成一份专业的脊柱侧弯诊断报告。

    返回一个生成器，每次产生报告的一个片段

    Args:
        patient_data (dict): 包含患者信息的字典。
                             必需的键: 'age', 'sex', 'cobb_angles'。
                             'cobb_angles' 本身应为一个字典。
                             可选的键: 'symptoms', 'skeletal_maturity', 'other_findings'

    Yields:
        str: 报告内容片段或错误信息
    """
    if not client:
        yield "错误: AI服务客户端未成功初始化。"
        return

    # --- 从输入字典中安全地提取信息 ---
    age = patient_data.get('age', '未提供')
    sex = patient_data.get('sex', '未提供')
    symptoms = patient_data.get('symptoms', '体检发现')
    skeletal_maturity = patient_data.get('skeletal_maturity', '未提供')
    other_findings = patient_data.get('other_findings', '无特殊记录')

    # 提取并格式化Cobb角信息
    cobb_angles_dict = patient_data.get('cobb_angles', {})
    if not cobb_angles_dict or not isinstance(cobb_angles_dict, dict):
        yield "错误: 输入数据中缺少或格式不正确的 'cobb_angles' 字典。"
        return

    cobb_angles_summary = "\n".join([
        f"    - 主胸弯 (PT): {cobb_angles_dict.get('主胸弯 (PT)', '未测量')}",
        f"    - 胸腰弯/腰弯 (TL/L): {cobb_angles_dict.get('胸腰弯/腰弯 (TL/L)', '未测量')}",
        f"    - 上胸弯 (MT): {cobb_angles_dict.get('上胸弯 (MT)', '未测量')}"
    ])

    # --- 构建发送给AI的提示 ---
    system_message = """
    你是一位资深骨科脊柱外科医生，拥有20年临床经验。请根据提供的影像学数据和患者信息，
    撰写一份专业、全面的脊柱侧弯诊断报告和治疗方案。报告需包含以下部分：
    1. 影像学发现总结
    2. 临床诊断
    3. 病情评估（包括进展风险分析）
    4. 详细治疗方案
    5. 随访计划
    使用专业医学术语，语言严谨准确，格式清晰。
    """

    user_content = f"""
    ## 患者基本信息
    - 年龄: {age}
    - 性别: {sex}
    - 主诉: {symptoms}

    ## 影像学检查结果
    - **Cobb角度**:
{cobb_angles_summary}
    - **骨骼成熟度**: {skeletal_maturity}
    - **其他发现**: {other_findings}

    ## 报告要求
    请根据以上信息，生成包含以下内容的完整临床报告：
    ### 1. 影像学发现总结
    ### 2. 临床诊断
    ### 3. 病情评估与风险分析（重点分析最大Cobb角的临床意义和进展风险）
    ### 4. 个体化治疗方案（根据Cobb角度、年龄和骨骼成熟度制定）
    ### 5. 随访计划
    """

    # --- 流式调用AI模型 ---
    try:
        response = client.chat.completions.create(
            model='deepseek-chat',
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=2000,
            stream=True  # 启用流式响应
        )

        # 流式处理响应
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        error_message = f"\n调用AI模型API时发生错误: {str(e)}"
        yield error_message


# --- 3. 示例用法（流式版本）---
if __name__ == "__main__":
    # 假设这是从您的Cobb角计算流程中得到的输出
    cobb_angle_results = {
        '主胸弯 (PT)': '55.0°',
        '胸腰弯/腰弯 (TL/L)': '31.5°',
        '上胸弯 (MT)': '22.0°'
    }

    # 构建完整的患者数据字典
    patient_info = {
        "age": 15,
        "sex": "女",
        "symptoms": "学校体检筛查发现姿态异常，双肩不等高。",
        "cobb_angles": cobb_angle_results,
        "skeletal_maturity": "Risser 2",
        "other_findings": "椎体存在轻度旋转，矢状面曲度基本正常。"
    }

    print("正在为患者生成AI诊断报告...")
    print("-" * 60)
    print("AI生成的脊柱侧弯诊断报告与治疗方案".center(60))
    print("-" * 60)

    # 流式生成报告
    start_time = time.time()
    report_chunks = stream_ai_report(patient_info)

    # 打印流式结果
    for chunk in report_chunks:
        print(chunk, end='', flush=True)  # 逐块打印，不换行
        time.sleep(0.02)  # 添加微小延迟使输出更流畅

    end_time = time.time()
    print("\n" + "-" * 60)
    print(f"报告生成完成，耗时: {end_time - start_time:.2f}秒")
    print("注意：本报告由AI辅助生成，仅供参考，最终诊断和治疗方案需由专业执业医师审核确认。")