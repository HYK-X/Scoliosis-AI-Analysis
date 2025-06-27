# # 文件名: cobb_calculator.py
# # -*- coding: utf-8 -*-
#
# import numpy as np
# import cv2
# import os
#
# # --- 新增：引入Pillow库用于中文文本绘制 ---
# try:
#     from PIL import Image, ImageDraw, ImageFont
# except ImportError:
#     print("警告: Pillow库未安装。中文文本将无法显示。")
#     print("建议安装: pip install Pillow")
#     Image, ImageDraw, ImageFont = None, None, None
#
# # 引入 seaborn 用于更美观的调色板
# try:
#     import seaborn as sns
# except ImportError:
#     print("警告: Seaborn库未安装。将使用备用颜色方案。")
#     print("建议安装: pip install seaborn")
#     sns = None
#
# # Matplotlib的设置在这里对最终的图片输出不起作用，因为我们使用OpenCV和Pillow绘制
# # 但保留它可能对未来直接使用matplotlib绘图有用
# import matplotlib.pyplot as plt
#
# try:
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.rcParams['axes.unicode_minus'] = False
# except:
#     print("警告: 设置中文字体失败，如果直接使用matplotlib绘图，标题可能显示为方框。")
#
# # --- 核心配置 ---
# NUM_KEYPOINTS_PER_GROUP = 4
# SKELETON = [[0, 1], [1, 2], [2, 3], [3, 0]]  # 椎体四个角的连接关系
#
# # --- 新增：定义中文字体路径 ---
# # !! 重要 !!: 请确保此路径指向一个有效的中文字体文件（.ttf 或 .otf）
# # 简单方法：将字体文件（如 SimHei.ttf）放在与此脚本相同的文件夹中。
# FONT_PATH = 'SimHei.ttf'
#
#
# # --- 辅助绘图函数 ---
#
# def _get_dynamic_sizes(image):
#     """根据图像高度动态计算绘图元素的尺寸，以保证视觉效果一致。"""
#     h, _, _ = image.shape
#     scale_factor = h / 1000.0  # 假设1000px是基准高度
#
#     sizes = {
#         'line_thick': max(1, int(2 * scale_factor)),
#         'main_line_thick': max(2, int(4 * scale_factor)),
#         'circle_radius': max(2, int(5 * scale_factor)),
#         'font_scale': max(0.5, 1.0 * scale_factor),  # <-- 已修复：将 scale_scale_factor 改为 scale_factor
#         'font_thick': max(1, int(2 * scale_factor))  # OpenCV备用
#     }
#     return sizes
#
#
# # --- 函数重写：使用Pillow支持中文 ---
# def _draw_text_with_background(image, text, position, font_scale, font_thick, text_color, bg_color):
#     """
#     在图像上绘制带有半透明背景的文本，使其更清晰。
#     此版本使用Pillow库以支持中文字符。
#     """
#     # 检查Pillow是否可用以及字体文件是否存在
#     if Image is None or not os.path.exists(FONT_PATH):
#         print(f"警告: Pillow未安装或字体文件 '{FONT_PATH}' 不存在。回退到OpenCV绘图，中文将不显示。")
#         # 回退到旧的OpenCV方法（不支持中文）
#         cv2.putText(image, "Font Error: Check Pillow/Font File", position, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
#                     text_color, font_thick)
#         return
#
#     x, y = position
#     font_size = int(30 * font_scale)  # 将font_scale转换为Pillow的字体大小（像素）
#     try:
#         font = ImageFont.truetype(FONT_PATH, font_size)
#     except IOError:
#         print(f"错误: 无法加载字体文件 '{FONT_PATH}'。请检查路径和文件完整性。")
#         return
#
#     # Pillow使用RGB颜色，而OpenCV使用BGR
#     text_color_rgb = (text_color[2], text_color[1], text_color[0])
#
#     # 将OpenCV图像转换为Pillow图像以进行文本绘制
#     # 创建一个临时的ImageDraw对象来计算文本尺寸
#     temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
#
#     lines = text.split('\n')
#     line_heights = []
#     max_width = 0
#
#     for line in lines:
#         # 使用 textbbox 获取更精确的边界框
#         try:
#             # Pillow 9.2.0+
#             bbox = temp_draw.textbbox((0, 0), line, font=font)
#             w = bbox[2] - bbox[0]
#             h = bbox[3] - bbox[1]
#         except AttributeError:
#             # Pillow < 9.2.0
#             w, h = temp_draw.textsize(line, font=font)
#
#         line_heights.append(h)
#         max_width = max(max_width, w)
#
#     # 计算背景框位置
#     margin = int(10 * font_scale)
#     total_height = sum(line_heights) + margin * (len(lines))
#     bg_x1, bg_y1 = x, y
#     bg_x2, bg_y2 = x + max_width + 2 * margin, y + total_height + margin
#
#     # 确保背景框在图像范围内
#     bg_y2 = min(bg_y2, image.shape[0])
#     bg_x2 = min(bg_x2, image.shape[1])
#
#     # 绘制半透明背景
#     sub_img = image[bg_y1:bg_y2, bg_x1:bg_x2]
#     white_rect = np.ones(sub_img.shape, dtype=np.uint8)
#     white_rect[:] = bg_color
#     res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
#     image[bg_y1:bg_y2, bg_x1:bg_x2] = res
#
#     # 转换整个图像以进行Pillow绘制
#     image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(image_pil)
#
#     # 绘制文本
#     current_y = y + margin
#     for i, line in enumerate(lines):
#         draw.text((x + margin, current_y), line, font=font, fill=text_color_rgb)
#         current_y += line_heights[i] + margin / len(lines)
#
#     # 将Pillow图像转换回OpenCV图像
#     # 使用numpy切片直接修改原图，避免返回新对象
#     image[:] = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
#
#
# # --- 核心计算与处理模块 (无改动) ---
#
# def is_s_curve(points):
#     """判断椎体中心点是否形成S型曲线。"""
#     num_points = points.shape[0]
#     if num_points < 3: return False
#     p1, p_last = points[0], points[-1]
#     if (p1[1] == p_last[1]) or (p1[0] == p_last[0]): return False
#     ll = [(p_i[1] - p_last[1]) / (p1[1] - p_last[1]) - (p_i[0] - p_last[0]) / (p1[0] - p_last[0]) for p_i in
#           points[1:-1]]
#     return not all(x >= 0 for x in ll) and not all(x <= 0 for x in ll)
#
#
# def calculate_cobb_angles(keypoints_reshaped):
#     """核心计算函数：根据重塑后的关键点计算Cobb角。"""
#     print("--- [模块B, 步骤1: 计算Cobb角] ---")
#     y_centers = keypoints_reshaped[:, :, 1].mean(axis=1)
#     sorted_indices = np.argsort(y_centers)
#     sorted_keypoints = keypoints_reshaped[sorted_indices]
#
#     p_coords = sorted_keypoints.reshape(-1, 2)
#     num_vertebrae = len(p_coords) // NUM_KEYPOINTS_PER_GROUP
#
#     if num_vertebrae < 2:
#         print("!! 警告: 椎骨数量不足 (<2)，无法计算Cobb角。")
#         return None
#
#     mid_p = np.zeros((num_vertebrae * 2, 2))
#     for n in range(num_vertebrae):
#         mid_p[n * 2, :] = (p_coords[n * 4 + 0, :] + p_coords[n * 4 + 1, :]) / 2
#         mid_p[n * 2 + 1, :] = (p_coords[n * 4 + 2, :] + p_coords[n * 4 + 3, :]) / 2
#
#     vec_m = np.array([mid_p[n * 2 + 1, :] - mid_p[n * 2, :] for n in range(num_vertebrae)])
#     norm_v = np.linalg.norm(vec_m, axis=1)
#     dot_v = np.dot(vec_m, vec_m.T)
#     norm_v_matrix = np.outer(norm_v, norm_v)
#     cosine_angles = np.divide(dot_v, norm_v_matrix, out=np.zeros_like(dot_v), where=norm_v_matrix != 0)
#     cosine_angles = np.clip(cosine_angles, -1.0, 1.0)
#     angles = np.arccos(cosine_angles)
#     np.fill_diagonal(angles, 0)
#
#     pt_rad = np.max(angles)
#     pt = np.rad2deg(pt_rad)
#     indices = np.unravel_index(np.argmax(angles), angles.shape)
#     pos2, pos1 = sorted(indices)
#
#     cob_angles = {'PT': pt, 'MT': 0.0, 'TL': 0.0}
#     mid_p_v = np.array([(mid_p[n * 2, :] + mid_p[n * 2 + 1, :]) / 2 for n in range(num_vertebrae)])
#
#     if not is_s_curve(mid_p_v):
#         vec_1, vec_last = vec_m[0, :], vec_m[-1, :]
#         vec_pos1, vec_pos2 = vec_m[pos1, :], vec_m[pos2, :]
#         mt_rad = np.arccos(np.clip(np.dot(vec_1, vec_pos2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_pos2)), -1, 1))
#         tl_rad = np.arccos(
#             np.clip(np.dot(vec_last, vec_pos1) / (np.linalg.norm(vec_last) * np.linalg.norm(vec_pos1)), -1, 1))
#         cob_angles['MT'], cob_angles['TL'] = np.rad2deg(mt_rad), np.rad2deg(tl_rad)
#
#     print("Cobb角计算完成。")
#     return cob_angles, mid_p, pos1, pos2
#
#
# def visualize_combined_results(image, keypoints_reshaped, cobb_results):
#     """核心可视化函数：将所有结果绘制到图像上，使用Seaborn风格。"""
#     print("--- [模块B, 步骤2: 生成融合可视化图像] ---")
#     vis_image = image.copy()
#     sizes = _get_dynamic_sizes(vis_image)
#
#     if sns:
#         vertebra_palette = sns.color_palette("viridis", n_colors=len(keypoints_reshaped))
#         colors = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255)) for c in vertebra_palette]
#         keypoint_color = (0, 255, 0)
#         midline_color = (255, 180, 200)
#         cobb_line_color = (0, 255, 255)
#         text_color = (255, 255, 255)
#         text_bg_color = (0, 0, 0)
#     else:
#         colors = [(255, 0, 0), (0, 0, 255), (0, 255, 255)]
#         keypoint_color, midline_color, cobb_line_color = (0, 255, 0), (150, 150, 255), (0, 255, 0)
#         text_color, text_bg_color = (255, 255, 0), (0, 0, 0)
#
#     for i, group in enumerate(keypoints_reshaped):
#         color = colors[i % len(colors)]
#         for p1_idx, p2_idx in SKELETON:
#             p1 = tuple(group[p1_idx].astype(int))
#             p2 = tuple(group[p2_idx].astype(int))
#             cv2.line(vis_image, p1, p2, color, sizes['line_thick'], cv2.LINE_AA)
#         for k_idx, (x, y) in enumerate(group):
#             cv2.circle(vis_image, (int(x), int(y)), sizes['circle_radius'], keypoint_color, -1, cv2.LINE_AA)
#
#     if cobb_results:
#         cob_angles, mid_p, pos1, pos2 = cobb_results
#
#         for n in range(len(mid_p) // 2):
#             p1 = tuple(mid_p[n * 2].astype(int))
#             p2 = tuple(mid_p[n * 2 + 1].astype(int))
#             cv2.line(vis_image, p1, p2, midline_color, sizes['line_thick'], cv2.LINE_AA)
#
#         def get_extended_line(p1, p2, length_factor=1.5):
#             center = (p1 + p2) / 2
#             d = p2 - p1
#             p_start = center - d * length_factor
#             p_end = center + d * length_factor
#             return tuple(p_start.astype(int)), tuple(p_end.astype(int))
#
#         p_upper_mid1, p_upper_mid2 = mid_p[pos2 * 2], mid_p[pos2 * 2 + 1]
#         p_lower_mid1, p_lower_mid2 = mid_p[pos1 * 2], mid_p[pos1 * 2 + 1]
#         start_upper, end_upper = get_extended_line(p_upper_mid1, p_upper_mid2)
#         start_lower, end_lower = get_extended_line(p_lower_mid1, p_lower_mid2)
#
#         cv2.line(vis_image, start_upper, end_upper, cobb_line_color, sizes['main_line_thick'], cv2.LINE_AA)
#         cv2.line(vis_image, start_lower, end_lower, cobb_line_color, sizes['main_line_thick'], cv2.LINE_AA)
#
#         # 改动：使用中文标签
#         angle_text = (f"上胸弯 (PT) Angle: {cob_angles['PT']:.1f}°\n"
#                       f"胸腰弯 (MT) Angle: {cob_angles['MT']:.1f}°\n"
#                       f"腰弯 (TL) Angle: {cob_angles['TL']:.1f}°")
#
#         # 此函数现在使用Pillow，可以正确显示中文
#         _draw_text_with_background(vis_image, angle_text, (15, 15),
#                                    sizes['font_scale'], sizes['font_thick'], text_color, text_bg_color)
#
#     return vis_image
#
#
# def process_image_with_points(image, keypoints):
#     """
#     主处理函数：接收图像和关键点，完成所有计算和可视化。
#     """
#     if not isinstance(image, np.ndarray) or not isinstance(keypoints, np.ndarray):
#         raise TypeError("输入图像和关键点必须是Numpy数组。")
#
#     if keypoints.ndim == 2:
#         if keypoints.shape[0] % NUM_KEYPOINTS_PER_GROUP != 0:
#             print(f"!! 错误: 关键点总数 ({keypoints.shape[0]}) 不是 {NUM_KEYPOINTS_PER_GROUP} 的倍数。")
#             # 当点数不匹配时，我们无法reshape成(N,4,2)，但仍可尝试可视化单个点
#             # 创建一个可以被visualize_combined_results处理的形状
#             keypoints_for_vis = np.array([kp for kp in keypoints]).reshape(-1, 1, 2)
#             final_image = visualize_combined_results(image, keypoints_for_vis, None)
#             return final_image, None
#         keypoints_reshaped = keypoints.reshape(-1, NUM_KEYPOINTS_PER_GROUP, 2)
#     elif keypoints.ndim == 3 and keypoints.shape[1:] == (NUM_KEYPOINTS_PER_GROUP, 2):
#         keypoints_reshaped = keypoints
#     else:
#         raise ValueError(f"关键点数组的形状不正确。实际形状: {keypoints.shape}")
#
#     cobb_results = calculate_cobb_angles(keypoints_reshaped)
#     final_image = visualize_combined_results(image, keypoints_reshaped, cobb_results)
#     angles_dict = cobb_results[0] if cobb_results else None
#
#     if angles_dict:
#         # 改动：使用中文标签
#         formatted_angles = {
#             f"上胸弯 (PT)": f"{angles_dict['PT']:.1f}°",
#             f"胸腰弯/腰弯 (TL/L)": f"{angles_dict['TL']:.1f}°",
#             f"主胸弯 (MT)": f"{angles_dict['MT']:.1f}°"
#         }
#     else:
#         formatted_angles = {"提示": "未能计算Cobb角"}
#
#     return final_image, formatted_angles
# 文件名: cobb_calculator.py
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os

# --- 新增：引入Pillow库用于中文文本绘制 ---
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("警告: Pillow库未安装。中文文本将无法显示。")
    print("建议安装: pip install Pillow")
    Image, ImageDraw, ImageFont = None, None, None

# 引入 seaborn 用于更美观的调色板
try:
    import seaborn as sns
except ImportError:
    print("警告: Seaborn库未安装。将使用备用颜色方案。")
    print("建议安装: pip install seaborn")
    sns = None

# Matplotlib的设置在这里对最终的图片输出不起作用，因为我们使用OpenCV和Pillow绘制
# 但保留它可能对未来直接使用matplotlib绘图有用
import matplotlib.pyplot as plt

try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("警告: 设置中文字体失败，如果直接使用matplotlib绘图，标题可能显示为方框。")

# --- 核心配置 ---
NUM_KEYPOINTS_PER_GROUP = 4
SKELETON = [[0, 1], [1, 2], [2, 3], [3, 0]]  # 椎体四个角的连接关系

# --- 新增：定义中文字体路径 ---
# !! 重要 !!: 请确保此路径指向一个有效的中文字体文件（.ttf 或 .otf）
# 简单方法：将字体文件（如 SimHei.ttf）放在与此脚本相同的文件夹中。
FONT_PATH = 'SimHei.ttf'


# --- 辅助绘图函数 (无改动) ---

def _get_dynamic_sizes(image):
    """根据图像高度动态计算绘图元素的尺寸，以保证视觉效果一致。"""
    h, _, _ = image.shape
    scale_factor = h / 1000.0  # 假设1000px是基准高度

    sizes = {
        'line_thick': max(1, int(2 * scale_factor)),
        'main_line_thick': max(2, int(4 * scale_factor)),
        'circle_radius': max(2, int(5 * scale_factor)),
        'font_scale': max(0.5, 1.0 * scale_factor),
        'font_thick': max(1, int(2 * scale_factor))  # OpenCV备用
    }
    return sizes


def _draw_text_with_background(image, text, position, font_scale, font_thick, text_color, bg_color):
    """
    在图像上绘制带有半透明背景的文本，使其更清晰。
    此版本使用Pillow库以支持中文字符。
    """
    if Image is None or not os.path.exists(FONT_PATH):
        print(f"警告: Pillow未安装或字体文件 '{FONT_PATH}' 不存在。回退到OpenCV绘图，中文将不显示。")
        cv2.putText(image, "Font Error: Check Pillow/Font File", position, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    text_color, font_thick)
        return

    x, y = position
    font_size = int(30 * font_scale)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except IOError:
        print(f"错误: 无法加载字体文件 '{FONT_PATH}'。请检查路径和文件完整性。")
        return

    text_color_rgb = (text_color[2], text_color[1], text_color[0])
    temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    lines = text.split('\n')
    line_heights = []
    max_width = 0

    for line in lines:
        try:
            bbox = temp_draw.textbbox((0, 0), line, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            w, h = temp_draw.textsize(line, font=font)
        line_heights.append(h)
        max_width = max(max_width, w)

    margin = int(10 * font_scale)
    total_height = sum(line_heights) + margin * (len(lines))
    bg_x1, bg_y1 = x, y
    bg_x2, bg_y2 = x + max_width + 2 * margin, y + total_height + margin
    bg_y2 = min(bg_y2, image.shape[0])
    bg_x2 = min(bg_x2, image.shape[1])

    sub_img = image[bg_y1:bg_y2, bg_x1:bg_x2]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8)
    white_rect[:] = bg_color
    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
    image[bg_y1:bg_y2, bg_x1:bg_x2] = res

    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    current_y = y + margin
    for i, line in enumerate(lines):
        draw.text((x + margin, current_y), line, font=font, fill=text_color_rgb)
        current_y += line_heights[i] + margin / len(lines)

    image[:] = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


# --- 核心计算与处理模块 (is_s_curve 已修改) ---

def is_s_curve(mid_points):
    """
    【已修改】判断椎体中心点是否形成S型曲线。
    此版本精确复现了原始MATLAB代码中“巧妙但不够直观”的矩阵运算法。
    """
    num_points = len(mid_points)
    if num_points < 3:
        return False  # 点太少，无法形成'S'形

    # 步骤 1: 复现 MATLAB 中的 linefun 逻辑，计算偏移量向量 ll
    p1 = mid_points[0]
    p_last = mid_points[-1]

    delta_y_total = p1[1] - p_last[1]
    delta_x_total = p1[0] - p_last[0]

    # 为防止除零错误，增加一个回退检查机制
    # 如果首尾点连线是纯粹的水平或垂直线
    if np.isclose(delta_y_total, 0) or np.isclose(delta_x_total, 0):
        # 使用更简单直接的方法判断
        if np.isclose(delta_x_total, 0):  # 垂直线
            deviations = mid_points[1:-1, 0] - p1[0]
        else:  # 水平线
            deviations = mid_points[1:-1, 1] - p1[1]

        has_positive = np.any(deviations > 1e-6)
        has_negative = np.any(deviations < -1e-6)
        return has_positive and has_negative

    # 计算偏移量向量 ll
    ll = []
    # 遍历所有中间点
    for i in range(1, num_points - 1):
        p_i = mid_points[i]
        # 这是 linefun 函数的核心公式
        val = (p_i[1] - p_last[1]) / delta_y_total - (p_i[0] - p_last[0]) / delta_x_total
        ll.append(val)

    ll = np.array(ll)

    # 如果没有中间点，则不是S型
    if ll.size == 0:
        return False

    # 步骤 2: 复现 isS 函数的核心矩阵运算逻辑
    # ll * ll' 在 MATLAB 中是向量的外积
    outer_product_matrix = np.outer(ll, ll)

    # sum(sum(matrix)) 在 numpy 中是 np.sum(matrix)
    sum_of_matrix = np.sum(outer_product_matrix)
    sum_of_abs_matrix = np.sum(np.abs(outer_product_matrix))

    # flag = sum_of_matrix ~= sum_of_abs_matrix
    # 使用 np.isclose 进行浮点数安全比较，如果不相等，则为S型
    return not np.isclose(sum_of_matrix, sum_of_abs_matrix)


def calculate_cobb_angles(keypoints_reshaped):
    """
    【已替换】核心计算函数：根据重塑后的关键点计算Cobb角。
    此版本采用了更精确的算法来区分和计算C/S型曲线。
    """
    print("--- [模块B, 步骤1: 计算Cobb角 (使用新算法)] ---")

    # 步骤 1: 按y轴坐标对椎体进行排序，确保从上到下的顺序
    y_centers = keypoints_reshaped[:, :, 1].mean(axis=1)
    sorted_indices = np.argsort(y_centers)
    sorted_keypoints = keypoints_reshaped[sorted_indices]

    p_coords = sorted_keypoints.reshape(-1, 2)
    num_vertebrae = len(p_coords) // NUM_KEYPOINTS_PER_GROUP

    if num_vertebrae < 2:
        print("!! 警告: 椎骨数量不足 (<2)，无法计算Cobb角。")
        return None

    # 步骤 2: 计算每个椎体的上、下缘中点
    mid_p = np.zeros((num_vertebrae * 2, 2))
    for n in range(num_vertebrae):
        # 上缘中点
        mid_p[n * 2, :] = (p_coords[n * 4 + 0, :] + p_coords[n * 4 + 1, :]) / 2
        # 下缘中点
        mid_p[n * 2 + 1, :] = (p_coords[n * 4 + 2, :] + p_coords[n * 4 + 3, :]) / 2

    # 步骤 3: 计算代表每个椎体倾斜度的向量 (从下缘中点指向上缘中点)
    vec_m = np.array([mid_p[n * 2, :] - mid_p[n * 2 + 1, :] for n in range(num_vertebrae)])

    # 步骤 4: 计算所有椎体间的角度，找到主Cobb角 (PT)
    norm_v = np.linalg.norm(vec_m, axis=1)
    # 防止除以零
    if np.any(norm_v == 0):
        print("!! 警告: 存在长度为零的椎体向量，无法计算。")
        return None

    dot_v = np.dot(vec_m, vec_m.T)
    norm_v_matrix = np.outer(norm_v, norm_v)

    # 使用 np.divide 安全除法
    cosine_angles = np.divide(dot_v, norm_v_matrix, out=np.zeros_like(dot_v), where=norm_v_matrix != 0)
    cosine_angles = np.clip(cosine_angles, -1.0, 1.0)
    angles_matrix = np.arccos(cosine_angles)
    np.fill_diagonal(angles_matrix, 0)  # 忽略自身与自身的夹角

    # 主Cobb角是所有角度中的最大值
    pt_rad = np.max(angles_matrix)
    pt = np.rad2deg(pt_rad)
    # 获取最大值所在的两个椎体的索引
    indices = np.unravel_index(np.argmax(angles_matrix), angles_matrix.shape)
    pos2, pos1 = sorted(indices)  # pos2是上方椎体，pos1是下方椎体

    # 步骤 5: 计算次级Cobb角 (MT, TL)，区分C型和S型曲线
    mt, tl = 0.0, 0.0
    # 计算用于判断S型的椎体中心点
    mid_p_v = np.array([(mid_p[n * 2, :] + mid_p[n * 2 + 1, :]) / 2 for n in range(num_vertebrae)])

    if not is_s_curve(mid_p_v):
        # C型曲线: 测量主曲线相对于最顶部和最底部椎体的角度
        # (这种定义较为罕见，但在此实现以匹配原始逻辑)
        vec_0, vec_last = vec_m[0, :], vec_m[-1, :]
        vec_pos1, vec_pos2 = vec_m[pos1, :], vec_m[pos2, :]
        mt_rad = np.arccos(np.clip(np.dot(vec_0, vec_pos2) / (np.linalg.norm(vec_0) * np.linalg.norm(vec_pos2)), -1, 1))
        tl_rad = np.arccos(
            np.clip(np.dot(vec_last, vec_pos1) / (np.linalg.norm(vec_last) * np.linalg.norm(vec_pos1)), -1, 1))
        mt, tl = np.rad2deg(mt_rad), np.rad2deg(tl_rad)
    else:
        # S型曲线: 分别计算上段和下段的最大Cobb角
        # 上段次级曲线 (MT): 从最顶椎到主曲线顶椎(pos2)
        if pos2 > 0:  # 确保上段至少有2个椎体
            upper_segment_vectors = vec_m[0:pos2 + 1, :]
            norm_upper = np.linalg.norm(upper_segment_vectors, axis=1)
            dot_upper = np.dot(upper_segment_vectors, upper_segment_vectors.T)
            norm_upper_matrix = np.outer(norm_upper, norm_upper)
            cos_upper = np.divide(dot_upper, norm_upper_matrix, out=np.zeros_like(dot_upper),
                                  where=norm_upper_matrix != 0)
            upper_angles = np.arccos(np.clip(cos_upper, -1, 1))
            np.fill_diagonal(upper_angles, 0)
            mt = np.rad2deg(np.max(upper_angles))

        # 下段次级曲线 (TL/L): 从主曲线底椎(pos1)到最底椎
        if pos1 < num_vertebrae - 1:  # 确保下段至少有2个椎体
            lower_segment_vectors = vec_m[pos1:, :]
            norm_lower = np.linalg.norm(lower_segment_vectors, axis=1)
            dot_lower = np.dot(lower_segment_vectors, lower_segment_vectors.T)
            norm_lower_matrix = np.outer(norm_lower, norm_lower)
            cos_lower = np.divide(dot_lower, norm_lower_matrix, out=np.zeros_like(dot_lower),
                                  where=norm_lower_matrix != 0)
            lower_angles = np.arccos(np.clip(cos_lower, -1, 1))
            np.fill_diagonal(lower_angles, 0)
            tl = np.rad2deg(np.max(lower_angles))

    # 步骤 6: 封装结果以匹配函数输出格式
    cob_angles = {'PT': pt, 'MT': mt, 'TL': tl}

    print("Cobb角计算完成。")
    # 返回与可视化函数兼容的元组
    return cob_angles, mid_p, pos1, pos2


def visualize_combined_results(image, keypoints_reshaped, cobb_results):
    """核心可视化函数：将所有结果绘制到图像上，使用Seaborn风格。(无改动)"""
    print("--- [模块B, 步骤2: 生成融合可视化图像] ---")
    vis_image = image.copy()
    sizes = _get_dynamic_sizes(vis_image)

    if sns:
        vertebra_palette = sns.color_palette("viridis", n_colors=len(keypoints_reshaped))
        colors = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255)) for c in vertebra_palette]
        keypoint_color = (0, 255, 0)
        midline_color = (255, 180, 200)
        cobb_line_color = (0, 255, 255)
        text_color = (255, 255, 255)
        text_bg_color = (0, 0, 0)
    else:
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 255)]
        keypoint_color, midline_color, cobb_line_color = (0, 255, 0), (150, 150, 255), (0, 255, 0)
        text_color, text_bg_color = (255, 255, 0), (0, 0, 0)

    for i, group in enumerate(keypoints_reshaped):
        color = colors[i % len(colors)]
        for p1_idx, p2_idx in SKELETON:
            p1 = tuple(group[p1_idx].astype(int))
            p2 = tuple(group[p2_idx].astype(int))
            cv2.line(vis_image, p1, p2, color, sizes['line_thick'], cv2.LINE_AA)
        for k_idx, (x, y) in enumerate(group):
            cv2.circle(vis_image, (int(x), int(y)), sizes['circle_radius'], keypoint_color, -1, cv2.LINE_AA)

    if cobb_results:
        cob_angles, mid_p, pos1, pos2 = cobb_results

        for n in range(len(mid_p) // 2):
            p1 = tuple(mid_p[n * 2].astype(int))
            p2 = tuple(mid_p[n * 2 + 1].astype(int))
            cv2.line(vis_image, p1, p2, midline_color, sizes['line_thick'], cv2.LINE_AA)

        def get_extended_line(p1, p2, length_factor=1.5):
            center = (p1 + p2) / 2
            d = p2 - p1
            p_start = center - d * length_factor
            p_end = center + d * length_factor
            return tuple(p_start.astype(int)), tuple(p_end.astype(int))

        p_upper_mid1, p_upper_mid2 = mid_p[pos2 * 2], mid_p[pos2 * 2 + 1]
        p_lower_mid1, p_lower_mid2 = mid_p[pos1 * 2], mid_p[pos1 * 2 + 1]
        start_upper, end_upper = get_extended_line(p_upper_mid1, p_upper_mid2)
        start_lower, end_lower = get_extended_line(p_lower_mid1, p_lower_mid2)

        cv2.line(vis_image, start_upper, end_upper, cobb_line_color, sizes['main_line_thick'], cv2.LINE_AA)
        cv2.line(vis_image, start_lower, end_lower, cobb_line_color, sizes['main_line_thick'], cv2.LINE_AA)

        angle_text = (f"上胸弯 (PT) Angle: {cob_angles['PT']:.1f}°\n"
                      f"主胸弯 (MT) Angle: {cob_angles['MT']:.1f}°\n"
                      f"胸腰弯 (TL) Angle: {cob_angles['TL']:.1f}°")

        _draw_text_with_background(vis_image, angle_text, (15, 15),
                                   sizes['font_scale'], sizes['font_thick'], text_color, text_bg_color)

    return vis_image


def process_image_with_points(image, keypoints):
    """
    主处理函数：接收图像和关键点，完成所有计算和可视化。(无改动)
    """
    if not isinstance(image, np.ndarray) or not isinstance(keypoints, np.ndarray):
        raise TypeError("输入图像和关键点必须是Numpy数组。")

    if keypoints.ndim == 2:
        if keypoints.shape[0] % NUM_KEYPOINTS_PER_GROUP != 0:
            print(f"!! 错误: 关键点总数 ({keypoints.shape[0]}) 不是 {NUM_KEYPOINTS_PER_GROUP} 的倍数。")
            keypoints_for_vis = np.array([kp for kp in keypoints]).reshape(-1, 1, 2)
            final_image = visualize_combined_results(image, keypoints_for_vis, None)
            return final_image, None
        keypoints_reshaped = keypoints.reshape(-1, NUM_KEYPOINTS_PER_GROUP, 2)
    elif keypoints.ndim == 3 and keypoints.shape[1:] == (NUM_KEYPOINTS_PER_GROUP, 2):
        keypoints_reshaped = keypoints
    else:
        raise ValueError(f"关键点数组的形状不正确。实际形状: {keypoints.shape}")

    cobb_results = calculate_cobb_angles(keypoints_reshaped)
    final_image = visualize_combined_results(image, keypoints_reshaped, cobb_results)
    angles_dict = cobb_results[0] if cobb_results else None

    if angles_dict:
        # 改动可视化文本以匹配新的MT定义
        formatted_angles = {
            f"上胸弯 (PT)": f"{angles_dict['PT']:.1f}°",
            f"主胸弯 (MT)": f"{angles_dict['MT']:.1f}°",
            f"胸腰弯/腰弯 (TL/L)": f"{angles_dict['TL']:.1f}°"
        }
    else:
        formatted_angles = {"提示": "未能计算Cobb角"}

    return final_image, formatted_angles