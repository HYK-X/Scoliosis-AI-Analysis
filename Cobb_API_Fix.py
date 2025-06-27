# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.interpolate import UnivariateSpline
import warnings
import seaborn as sns  # 引入seaborn用于高级配色

warnings.filterwarnings('ignore')

# 尝试设置中文字体，如果失败则发出警告
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("警告: 设置中文字体失败，Matplotlib标题可能显示为方框。")

# --- 核心配置与常量 ---
NUM_KEYPOINTS_PER_GROUP = 4  # 每组关键点（一个椎骨）的数量
SKELETON = [[0, 1], [1, 2], [2, 3], [3, 0]]  # 椎骨关键点的连接顺序


class CobbAngleCalculator:
    """
    增强版Cobb角计算器，包含多种实现策略
    - 端板法（固定组合）
    - 中心线法（最大切线夹角）
    """

    def __init__(self, method='endplate'):
        """
        初始化计算器

        Args:
            method (str): 计算方法 ('endplate' 或 'centerline')
        """
        self.method = method

    # ------------------------------------------------------------------
    # **端板法**
    # ------------------------------------------------------------------
    def calculate_endplate_cobb_angles(self, keypoints_reshaped):
        print("--- 使用基于可视化线段的端板法 Cobb 角计算方法 ---")

        # 按 Y 坐标排序，确保自上而下顺序一致
        y_centers = keypoints_reshaped[:, :, 1].mean(axis=1)
        sorted_indices = np.argsort(y_centers)
        sorted_keypoints = keypoints_reshaped[sorted_indices]
        num_vertebrae = len(sorted_keypoints)

        if num_vertebrae < 16:
            print(f"!! 警告: 椎骨数量不足 ({num_vertebrae} < 16)，无法按照固定组合计算。")
            return None, None, None

        # 构建可视化用的端板线段列表
        endplates = {'upper': [], 'lower': []}
        for v in sorted_keypoints:
            endplates['upper'].append((v[0], v[1]))  # 上端板: UL->UR
            endplates['lower'].append((v[3], v[2]))  # 下端板: LL->LR

        # 计算三段 Cobb 角（PT, MT, TL）
        angles_details = self._calculate_fixed_cobb_angles(endplates)
        return angles_details, endplates, None


    # ------------------------------------------------------------------
    # **中心线法**
    # ------------------------------------------------------------------
    def calculate_centerline_cobb_angles(self, keypoints_reshaped):
        """中心线拟合法 Cobb 角计算（仅返回主胸弯 MT，其余为 0）"""
        print("--- 使用简化版中心线 Cobb 角计算方法 (最大切线夹角) ---")

        center_points = np.mean(keypoints_reshaped, axis=1)
        sorted_indices = np.argsort(center_points[:, 1])
        sorted_centers = center_points[sorted_indices]
        num_points = len(sorted_centers)

        if num_points < 5:
            print("!! 警告: 椎骨数量不足 (<5)，无法进行曲线拟合。")
            return None, None, None

        try:
            x, y = sorted_centers[:, 0], sorted_centers[:, 1]
            spline_x = UnivariateSpline(y, x, s=len(y))
            y_dense = np.linspace(y.min(), y.max(), num_points * 10)
            x_dense = spline_x(y_dense)
            dx_dy = spline_x.derivative()(y_dense)
            theta = np.arctan(dx_dy)
            idx_max, idx_min = np.argmax(theta), np.argmin(theta)
            mt_angle_rad = abs(theta[idx_max] - theta[idx_min])
            mt_angle_deg = np.rad2deg(mt_angle_rad)

            angles_dict = {'PT': 0.0, 'TL': 0.0, 'MT': mt_angle_deg}
            centerline_data = {
                'original_centers': sorted_centers,
                'fitted_x': x_dense, 'fitted_y': y_dense,
                'slopes': dx_dy, 'theta': theta,
                'idx_max_theta': idx_max, 'idx_min_theta': idx_min
            }
            extra_info = {
                'theta_max_deg': np.rad2deg(theta[idx_max]),
                'theta_min_deg': np.rad2deg(theta[idx_min])
            }
            return angles_dict, centerline_data, extra_info
        except Exception as e:
            print(f"!! 错误: 曲线拟合失败 - {e}")
            return None, None, None

    def _calculate_fixed_cobb_angles(self, endplates):
        """按照固定椎体组合 (1-6, 6-12, 12-16) 计算 Cobb 角，使用可视化线段确保一致性"""
        mapping = {
            'PT': (0, 5),   # Proximal Thoracic: 1–6
            'MT': (5, 11),  # Main Thoracic: 6–12
            'TL': (11, 15)  # Thoraco-Lumbar: 12–16
        }
        angles_details = {}

        for name, (u_idx, l_idx) in mapping.items():
            # 如果索引超出范围，则跳过
            if u_idx >= len(endplates['upper']) or l_idx >= len(endplates['lower']):
                angles_details[name] = None
                continue

            # 取出两条可视化端板线段
            p1, p2 = map(np.array, endplates['upper'][u_idx])
            p3, p4 = map(np.array, endplates['lower'][l_idx])

            # 计算向量
            v1 = p2 - p1
            v2 = p4 - p3

            # 用点积求夹角，并确保是锐角
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            # 数值稳定性处理
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            # 如果大于 90°，取补角
            if theta > np.pi / 2:
                theta = np.pi - theta

            angles_details[name] = {
                'angle': np.rad2deg(theta),
                'upper_idx': u_idx,
                'lower_idx': l_idx
            }

        return angles_details

# --- 几何辅助函数 ---
def _solve_line_intersection(p1, p2, p3, p4):
    """求解两条线段所在直线的交点。"""
    v1 = p2 - p1
    v2 = p4 - p3
    p1p3 = p3 - p1
    den = v1[0] * (-v2[1]) - (-v2[0]) * v1[1]
    if abs(den) < 1e-6: return None, None, None
    t = (p1p3[0] * (-v2[1]) - (-v2[0]) * p1p3[1]) / den
    u = (v1[0] * p1p3[1] - v1[1] * p1p3[0]) / den
    return p1 + t * v1, t, u


# --- 可视化模块 ---

def draw_angle_lines(img, line1_pts, line2_pts, angle_name, angle_value, color, text_pos=(0, 0)):
    """在图像上绘制Cobb角线、延长线和文本标签(坐标已偏移)"""
    line1 = np.array(line1_pts)
    line2 = np.array(line2_pts)
    cv2.line(img, tuple(line1[0].astype(int)), tuple(line1[1].astype(int)), color, 4)
    cv2.line(img, tuple(line2[0].astype(int)), tuple(line2[1].astype(int)), color, 4)

    intersection, t, u = _solve_line_intersection(line1[0], line1[1], line2[0], line2[1])

    if intersection is not None:
        if t > 1:
            start_pt_1 = line1[1]
        elif t < 0:
            start_pt_1 = line1[0]
        else:
            start_pt_1 = None
        if start_pt_1 is not None:
            cv2.line(img, tuple(start_pt_1.astype(int)), tuple(intersection.astype(int)), color, 4, cv2.LINE_AA)

        if u > 1:
            start_pt_2 = line2[1]
        elif u < 0:
            start_pt_2 = line2[0]
        else:
            start_pt_2 = None
        if start_pt_2 is not None:
            cv2.line(img, tuple(start_pt_2.astype(int)), tuple(intersection.astype(int)), color, 4, cv2.LINE_AA)

    text = f"{angle_name}: {angle_value:.1f} deg"
    cv2.putText(img, text, (int(text_pos[0]), int(text_pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def visualize_enhanced_results(image, keypoints_reshaped, calculation_results, method):
    """
    【v4 修复版】可视化函数，通过动态定界确保所有交点可见。
    """
    print(f"--- 生成动态定界的可视化图像 ({method} 方法) ---")
    h, w, _ = image.shape
    # 强制使用白色背景
    background_color = (255, 255, 255)

    if not calculation_results or not calculation_results[0]:
        vis_image = image.copy()
        cv2.putText(vis_image, "计算失败", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return vis_image

    angles_details, extra_data, _ = calculation_results

    # --- 1. 预计算所有需要显示的点，以确定画布边界 ---
    all_points_to_render = list(keypoints_reshaped.reshape(-1, 2))
    intersections = {}  # 用字典存储交点，便于后续使用

    if method == 'endplate':
        for name, details in angles_details.items():
            if details:
                line1 = extra_data['upper'][details['upper_idx']]
                line2 = extra_data['lower'][details['lower_idx']]
                intersect, _, _ = _solve_line_intersection(np.array(line1[0]), np.array(line1[1]), np.array(line2[0]),
                                                           np.array(line2[1]))
                if intersect is not None:
                    all_points_to_render.append(intersect)
                    intersections[name] = intersect

    elif method == 'centerline':
        all_points_to_render.extend(np.column_stack([extra_data['fitted_x'], extra_data['fitted_y']]))
        s1 = extra_data['slopes'][extra_data['idx_max_theta']]
        s2 = extra_data['slopes'][extra_data['idx_min_theta']]
        p1 = np.array(
            [extra_data['fitted_x'][extra_data['idx_max_theta']], extra_data['fitted_y'][extra_data['idx_max_theta']]])
        p2 = np.array(
            [extra_data['fitted_x'][extra_data['idx_min_theta']], extra_data['fitted_y'][extra_data['idx_min_theta']]])
        if abs(s1 - s2) > 1e-6:
            y_i = (s1 * p1[1] - s2 * p2[1] + p2[0] - p1[0]) / (s1 - s2)
            x_i = p1[0] + s1 * (y_i - p1[1])
            intersect = np.array([x_i, y_i])
            all_points_to_render.append(intersect)
            intersections['PT'] = intersect

    # --- 2. 计算新画布尺寸和偏移量 ---
    all_points_np = np.array(all_points_to_render)
    min_coords = all_points_np.min(axis=0)
    max_coords = all_points_np.max(axis=0)

    final_min = np.minimum(min_coords, [0, 0])
    final_max = np.maximum(max_coords, [w, h])

    padding = 50  # 像素边距
    new_w = int(final_max[0] - final_min[0] + 2 * padding)
    new_h = int(final_max[1] - final_min[1] + 2 * padding)
    offset = -final_min + padding

    # --- 3. 创建新画布并绘制所有内容 ---
    vis_image = np.full((new_h, new_w, 3), background_color, dtype=np.uint8)
    vis_image[int(offset[1]):int(offset[1]) + h, int(offset[0]):int(offset[0]) + w] = image.copy()

    keypoints_offset = keypoints_reshaped + offset
    vertebra_colors = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255)) for c in
                       sns.color_palette("viridis", len(keypoints_offset))]

    for i, group in enumerate(keypoints_offset):
        color_bgr = vertebra_colors[i]
        for p1_idx, p2_idx in SKELETON:
            cv2.line(vis_image, tuple(group[p1_idx].astype(int)), tuple(group[p2_idx].astype(int)), color_bgr, 4)
        for x, y in group:
            cv2.circle(vis_image, (int(x), int(y)), 4, (0, 255, 0), -1)

    angle_colors = {'PT': (0, 0, 255), 'TL': (0, 255, 0), 'MT': (255, 0, 0)}

    if method == 'endplate':
        endplates_offset = {'upper': [np.array(l) + offset for l in extra_data['upper']],
                            'lower': [np.array(l) + offset for l in extra_data['lower']]}

        for line in endplates_offset['upper']: cv2.line(vis_image, tuple(line[0].astype(int)),
                                                        tuple(line[1].astype(int)), (128, 128, 128), 4)
        for line in endplates_offset['lower']: cv2.line(vis_image, tuple(line[0].astype(int)),
                                                        tuple(line[1].astype(int)), (128, 128, 128), 4)

        text_y_start = padding + 20
        text_x_start = padding + 20
        for i, (name, details) in enumerate(angles_details.items()):
            if details:
                line1 = endplates_offset['upper'][details['upper_idx']]
                line2 = endplates_offset['lower'][details['lower_idx']]
                color = angle_colors.get(name, (255, 255, 255))
                text_pos = (text_x_start + i * 180, text_y_start)
                draw_angle_lines(vis_image, line1, line2, name, details['angle'], color, text_pos=text_pos)

    elif method == 'centerline':
        centers_offset = extra_data['original_centers'] + offset
        fitted_points_offset = np.column_stack([extra_data['fitted_x'] + offset[0], extra_data['fitted_y'] + offset[1]])

        for point in centers_offset: cv2.circle(vis_image, tuple(point.astype(int)), 6, (255, 100, 100), -1)
        cv2.polylines(vis_image, [fitted_points_offset.astype(int)], isClosed=False, color=(0, 255, 0), thickness=4,
                      lineType=cv2.LINE_AA)

        pt_max_offset = np.array([extra_data['fitted_x'][extra_data['idx_max_theta']] + offset[0],
                                  extra_data['fitted_y'][extra_data['idx_max_theta']] + offset[1]])
        pt_min_offset = np.array([extra_data['fitted_x'][extra_data['idx_min_theta']] + offset[0],
                                  extra_data['fitted_y'][extra_data['idx_min_theta']] + offset[1]])
        intersection_pt_offset = intersections.get('PT') + offset if intersections.get('PT') is not None else None

        color = angle_colors['PT']
        cv2.circle(vis_image, tuple(pt_max_offset.astype(int)), 10, color, -1)
        cv2.circle(vis_image, tuple(pt_min_offset.astype(int)), 10, color, -1)

        if intersection_pt_offset is not None:
            cv2.line(vis_image, tuple(pt_max_offset.astype(int)), tuple(intersection_pt_offset.astype(int)), color, 4,
                     cv2.LINE_AA)
            cv2.line(vis_image, tuple(pt_min_offset.astype(int)), tuple(intersection_pt_offset.astype(int)), color, 4,
                     cv2.LINE_AA)

        text = f"MT (Max Angle Diff): {angles_details['MT']:.1f} deg"
        cv2.putText(vis_image, text, (padding + 20, padding + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return vis_image


def process_image_with_enhanced_calculation(image, keypoints, method='endplate'):
    """主处理流程"""
    if not isinstance(image, np.ndarray) or not isinstance(keypoints, np.ndarray):
        raise TypeError("输入图像和关键点必须是Numpy数组。")

    if keypoints.ndim == 2:
        if keypoints.shape[0] % NUM_KEYPOINTS_PER_GROUP != 0:
            print(f"!! 错误: 关键点总数 ({keypoints.shape[0]}) 不是 {NUM_KEYPOINTS_PER_GROUP} 的倍数。")
            return image, None
        keypoints_reshaped = keypoints.reshape(-1, NUM_KEYPOINTS_PER_GROUP, 2)
    elif keypoints.ndim == 3 and keypoints.shape[1:] == (NUM_KEYPOINTS_PER_GROUP, 2):
        keypoints_reshaped = keypoints
    else:
        raise ValueError("关键点数组的形状不正确。")

    calculator = CobbAngleCalculator(method=method)
    calculation_results = None

    if method == 'endplate':
        calculation_results = calculator.calculate_endplate_cobb_angles(keypoints_reshaped)
    else:
        calculation_results = calculator.calculate_centerline_cobb_angles(keypoints_reshaped)

    final_image = visualize_enhanced_results(image, keypoints_reshaped, calculation_results, method)

    angles_dict = {}
    if calculation_results and calculation_results[0]:
        if method == 'endplate':
            for name, details in calculation_results[0].items():
                angles_dict[name] = details['angle'] if details else 0.0
        else:
            angles_dict = calculation_results[0]

    return final_image, angles_dict


def demo_cobb_calculation():
    """演示函数"""
    print("=== Cobb角计算器演示 (v4 - 动态定界修复版) ===")

    np.random.seed(43)  # 更换随机种子以产生更具挑战性的几何形状
    num_vertebrae = 17
    keypoints = []

    # 模拟一个更夸张的S型弯曲，以更好地测试交点在外部的情况
    center_x = 350
    for i in range(num_vertebrae):
        base_y = 60 + i * 35
        # 增加弯曲和倾斜的幅度，使得某些端板线更接近平行
        x_offset = 90 * np.sin(i / num_vertebrae * np.pi * 1.6) - 35 * np.sin(i / num_vertebrae * np.pi * 0.6)
        base_x = center_x + x_offset
        tilt_angle = (x_offset - (90 * np.sin((i - 1) / num_vertebrae * np.pi * 1.6) - 35 * np.sin(
            (i - 1) / num_vertebrae * np.pi * 0.6))) * 0.015

        width, height = 45, 22
        ul, ur, lr, ll = np.array([-width / 2, -height / 2]), np.array([width / 2, -height / 2]), np.array(
            [width / 2, height / 2]), np.array([-width / 2, height / 2])

        rot_matrix = np.array([[np.cos(tilt_angle), -np.sin(tilt_angle)], [np.sin(tilt_angle), np.cos(tilt_angle)]])
        vertebra_points = [np.dot(rot_matrix, p) + np.array([base_x, base_y]) for p in [ul, ur, lr, ll]]
        for point in vertebra_points: point += np.random.normal(0, 1.5, 2)
        keypoints.extend(vertebra_points)
    keypoints = np.array(keypoints)

    image_height, image_width = 800, 700
    background_color = (255, 255, 255)
    image = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)

    print(f"生成了 {num_vertebrae} 个椎骨的关键点数据，形状: {keypoints.shape}")
    methods = ['endplate', 'centerline']
    results_to_show = {}

    for method in methods:
        print(f"\n--- 测试 {method} 方法 ---")
        try:
            processed_image, angles_dict = process_image_with_enhanced_calculation(image.copy(), keypoints,
                                                                                   method=method)
            if angles_dict:
                print("计算结果:")
                for name, value in angles_dict.items():
                    if value is not None and value > 0: print(f"  {name}: {value:.1f}°")
                results_to_show[method] = processed_image
            else:
                print("计算失败")
                results_to_show[method] = image
        except Exception as e:
            print(f"处理过程中发生错误: {e}")
            results_to_show[method] = image

    fig, axes = plt.subplots(1, 2, figsize=(18, 12))
    fig.suptitle('Cobb角计算与可视化演示图', fontsize=16)

    if 'endplate' in results_to_show:
        axes[0].imshow(cv2.cvtColor(results_to_show['endplate'], cv2.COLOR_BGR2RGB))
        axes[0].set_title('端板法')
        axes[0].axis('off')

    if 'centerline' in results_to_show:
        axes[1].imshow(cv2.cvtColor(results_to_show['centerline'], cv2.COLOR_BGR2RGB))
        axes[1].set_title('中心线法 ')
        axes[1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    demo_cobb_calculation()