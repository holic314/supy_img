# 我有一个json文件，是coco格式的
#
# 1，其中有两个类别categories，类别2是分割数据，类别16是关键点数据。
#
# 2，其中json有三个数组，一是images，一个是categories，一个是annotations。
#
# 3，images中有一个id是图片id，annotations中的image_id是图片id。其中category_id为2的segmentation为眼肌面积的分割区域。其中category_id为16的keypoints为背膘厚关键点。
#
# 4，其中还有一个问题，segmentation可能有2个，但是只有不为空的那个是真数据。另外keypoints有七个点数据的才是真的关键点，然后第四个点才是我需要的点。
#
# 5，我还需要你帮我输出最后你检测的结果，需要输出：num,Image,Mask_list,LinePoint1,LinePoint2,Line_length,BCpoint1,BCpoint2,BMF
#
# 6，num是序号，从1开始计数。Image是文件名。Mask_list是segmentation坐标数据。LinePoint1和LinePoint2的计算方式是，计算眼肌的最大深度，也就是眼肌面积区域内最大的深度，不只是在坐标点上，而是区域上，面积中最大的深度，坐标应该是按照左上角为原点的，我需要的是平行为y轴的那个最大深度，需要计算出最大深度线段的上下两个点，即为LinePoint1,LinePoint2，同时需要得出他们之间的像素点长度Line_length。BCpoint1是keypoints数组七个点中的第四个点。BCpoint2是BCpoint1垂直向下相交于眼肌面积区域的第一个点的坐标。BMD是BCpoint1与BCpoint2的线段长度，用像素为单位。

import json
import numpy as np
import cv2 # OpenCV for polygon rasterization and mask operations
import math
import csv
from collections import defaultdict
import os

# --- 新增：计算多边形面积的函数 ---
def polygon_area(coords):
    """计算多边形的面积，使用鞋带定理

    Args:
        coords (list): 坐标点列表 [(x1, y1), (x2, y2), ...]

    Returns:
        float: 多边形的面积
    """
    n = len(coords)
    if n < 3: # 一个有效的多边形至少需要3个点
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]  # 下一个点，最后一个点连接到第一个点
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0
# --- 结束新增部分 ---

def calculate_max_vertical_depth(segmentation, img_height, img_width):
    """
    计算分割区域内垂直方向的最大深度及其端点。
    (函数内容保持不变，此处省略以保持简洁)
    """
    if not segmentation or not segmentation[0]:
        return None, None, 0

    # 1. 将 COCO 分割格式转换为 OpenCV fillPoly 格式
    points = np.array(segmentation[0]).reshape(-1, 2).astype(np.int32)
    if points.shape[0] < 3: # 需要至少3个点才能形成多边形
         print(f"警告: 分割数据点数少于3 ({points.shape[0]}个点)，无法计算深度。")
         return None, None, 0
    contours = [points.reshape((-1, 1, 2))]

    # 2. 创建一个空白掩码图
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # 3. 在掩码图上绘制填充的多边形
    try:
        cv2.fillPoly(mask, contours, 1) # 用 1 填充区域
    except Exception as e:
        print(f"警告: 使用 fillPoly 绘制分割时出错: {e}。分割点: {points.tolist()}")
        return None, None, 0

    # 4. 计算最大深度
    max_depth = 0
    best_x = -1
    best_y_top = -1
    best_y_bottom = -1

    try:
        x, y, w, h = cv2.boundingRect(points)
        if w <= 0 or h <= 0: # 检查宽度和高度是否有效
             print(f"警告: 无效的分割边界框 w={w}, h={h}，跳过最大深度计算。")
             return None, None, 0
        x_start, x_end = x, x + w
        y_start, y_end = y, y + h
    except Exception as e:
         print(f"警告: 计算边界框时出错 {e}，将扫描整个图像宽度。")
         x_start, x_end = 0, img_width
         y_start, y_end = 0, img_height # Fallback

    x_start = max(0, x_start)
    x_end = min(img_width, x_end)
    y_start = max(0, y_start)
    y_end = min(img_height, y_end)

    found_valid_column = False
    for x_coord in range(x_start, x_end):
        column_data = mask[y_start:y_end, x_coord]
        y_coords = np.where(column_data == 1)[0]

        if y_coords.size > 0:
            found_valid_column = True
            current_y_top = y_coords.min() + y_start
            current_y_bottom = y_coords.max() + y_start
            current_depth = current_y_bottom - current_y_top + 1

            if current_depth > max_depth:
                max_depth = current_depth
                best_x = x_coord
                best_y_top = current_y_top
                best_y_bottom = current_y_bottom

    if found_valid_column and best_x != -1:
        line_point1 = (best_x, best_y_top)
        line_point2 = (best_x, best_y_bottom)
        # Ensure depth is at least 1 if points are found
        calculated_depth = max(1, max_depth) if max_depth > 0 else 0
        return line_point1, line_point2, calculated_depth
    else:
        print(f"警告: 在分割区域内未找到任何有效的垂直深度。")
        return None, None, 0


def find_vertical_intersection(bc_point1, segmentation, img_height, img_width):
    """
    从 BCpoint1 垂直向下查找与分割区域的第一个交点。
    (函数内容保持不变，此处省略以保持简洁)
    """
    if not segmentation or not segmentation[0] or bc_point1 is None:
        return None, 0

    x_bc1, y_bc1 = bc_point1
    x_bc1_int = int(round(x_bc1)) # 使用四舍五入的整数x坐标进行掩码查找

    if not (0 <= x_bc1_int < img_width):
         print(f"警告: BCpoint1 的 x 坐标 ({x_bc1}) 超出图像宽度 ({img_width})。")
         return None, 0

    # 1. 创建掩码图
    points = np.array(segmentation[0]).reshape(-1, 2).astype(np.int32)
    if points.shape[0] < 3:
        print(f"警告: 分割数据点数少于3 ({points.shape[0]}个点)，无法计算交点。")
        return None, 0
    contours = [points.reshape((-1, 1, 2))]
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    try:
        cv2.fillPoly(mask, contours, 1)
    except Exception as e:
        print(f"警告: 使用 fillPoly 绘制分割时出错: {e}。分割点: {points.tolist()}")
        return None, 0


    # 2. 从 y_bc1 开始向下查找
    start_y = int(math.ceil(y_bc1))
    start_y = max(0, start_y)

    # 检查 BCpoint1 是否已经在分割区域内部或边界上
    # 需要小心浮点数精度问题，检查 start_y 是否有效且在该列的掩码内
    if start_y < img_height and mask[start_y, x_bc1_int] == 1:
        # 如果起始点就在区域内，直接返回该点？还是找边界？
        # 题目要求是“相交于眼肌面积区域的第一个点的坐标”
        # 如果BCpoint1就在区域内，那第一个“相交”点就是它自己下方紧邻的区域点
        # 我们保持原逻辑：从 start_y 开始向下找第一个 mask 为 1 的点
         pass # 继续下面的循环查找


    for y_coord in range(start_y, img_height):
        if mask[y_coord, x_bc1_int] == 1:
            bc_point2 = (x_bc1, y_coord) # Use original float x_bc1
            bmf = max(0, y_coord - y_bc1) # BMF should be non-negative
            return bc_point2, bmf

    print(f"警告: 在图像 {img_height}x{img_width} 的 x={x_bc1_int} 列，从 y={start_y} 开始向下未找到分割区域的交点。")
    return None, 0


def process_coco_json(json_path, output_csv_path):
    """
    主处理函数。
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到 JSON 文件：{json_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：无法解析 JSON 文件：{json_path}")
        return
    except Exception as e:
        print(f"读取 JSON 文件时发生错误：{e}")
        return

    images = {img['id']: img for img in coco_data.get('images', [])}
    annotations = coco_data.get('annotations', [])

    annotations_by_image = defaultdict(lambda: {'segmentation': [], 'keypoints': []})
    for ann in annotations:
        img_id = ann.get('image_id')
        cat_id = ann.get('category_id')
        if img_id not in images:
             print(f"警告: 标注 {ann.get('id')} 的 image_id {img_id} 在 images 列表中未找到，已跳过。")
             continue

        if cat_id == 2: # 眼肌分割
            seg = ann.get('segmentation')
            # 检查 seg 是否存在，是列表，非空，且第一个元素（轮廓）非空且至少有6个值（3个点）
            if seg and isinstance(seg, list) and seg[0] and isinstance(seg[0], list) and len(seg[0]) >= 6:
                annotations_by_image[img_id]['segmentation'].append(ann)
            else:
                 print(f"信息: 图像 {img_id} 的分割标注 {ann.get('id')} 为空、格式无效或点数少于3，已忽略。 Seg: {seg}")
        elif cat_id == 16: # 背膘关键点
            kpts = ann.get('keypoints')
            if kpts and isinstance(kpts, list) and len(kpts) == 7 * 3:
                annotations_by_image[img_id]['keypoints'].append(ann)
            else:
                 print(f"信息: 图像 {img_id} 的关键点标注 {ann.get('id')} 点数不为7 ({len(kpts)//3 if kpts else 0}个)，已忽略。")

    results = []
    num_counter = 0

    for img_id, anns in annotations_by_image.items():
        # 取第一个有效的分割和关键点标注
        valid_seg_ann = next((s for s in anns['segmentation'] if s.get('segmentation') and s['segmentation'][0] and len(s['segmentation'][0])>=6), None)
        valid_kpt_ann = next((k for k in anns['keypoints'] if k.get('keypoints') and len(k['keypoints']) == 21), None)

        if valid_seg_ann and valid_kpt_ann:
            num_counter += 1
            image_info = images[img_id]
            img_filename = image_info.get('file_name', f'Image_{img_id}')
            img_height = image_info.get('height')
            img_width = image_info.get('width')

            if img_height is None or img_width is None:
                 print(f"错误：图像 {img_id} ({img_filename}) 缺少 height 或 width 信息，无法处理。")
                 continue

            # --- 提取数据 ---
            mask_list = valid_seg_ann['segmentation'] # [[x1, y1, x2, y2, ...]]
            keypoints = valid_kpt_ann['keypoints']
            bc_point1_x = keypoints[9]
            bc_point1_y = keypoints[10]
            bc_point1 = (bc_point1_x, bc_point1_y)

            # --- 计算 ---
            # 0. 计算面积
            area = 0.0
            mask_coords_flat = mask_list[0]
            if len(mask_coords_flat) >= 6 and len(mask_coords_flat) % 2 == 0:
                try:
                    polygon_points = [(mask_coords_flat[i], mask_coords_flat[i+1]) for i in range(0, len(mask_coords_flat), 2)]
                    area = polygon_area(polygon_points)
                except Exception as e:
                    print(f"错误：计算图像 {img_filename} 的面积时出错: {e}")
                    area = 0.0 # 保持 area 为 0
            else:
                # 这个检查理论上在筛选时已经做过，但为了安全再加一层
                print(f"警告: 图像 {img_filename} 用于计算面积的分割坐标无效。坐标: {mask_coords_flat}")
                area = 0.0

            # 1. 计算最大深度
            line_point1, line_point2, line_length = calculate_max_vertical_depth(
                mask_list, img_height, img_width
            )

            # 2. 计算 BCpoint2 和 BMF
            bc_point2, bmf = find_vertical_intersection(
                bc_point1, mask_list, img_height, img_width
            )

            # --- 准备输出行 ---
            mask_list_str = str(mask_list) # 或者 json.dumps(mask_list)
            lp1_str = str(line_point1) if line_point1 else "None"
            lp2_str = str(line_point2) if line_point2 else "None"
            bcp1_str = str(bc_point1) if bc_point1 else "None"
            bcp2_str = str(bc_point2) if bc_point2 else "None"

            # --- 修改：添加 area 到结果列表 ---
            results.append([
                num_counter,
                img_filename,
                mask_list_str,
                area, # 新增面积列
                lp1_str,
                lp2_str,
                line_length if line_length is not None else 0,
                bcp1_str,
                bcp2_str,
                bmf if bmf is not None else 0
            ])
            print(f"处理完成: 图像 {img_filename} (序号 {num_counter}), 面积: {area:.2f}")


    # --- 输出到 CSV ---
    if not results:
        print("未找到任何同时包含有效分割和关键点标注的图像。")
        return

    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # --- 修改：添加 area 到表头 ---
            writer.writerow(['num', 'Image', 'Mask_list', 'area', 'LinePoint1', 'LinePoint2', 'Line_length', 'BCpoint1', 'BCpoint2', 'BMF'])
            writer.writerows(results)
        print(f"\n处理完成！结果已保存到：{output_csv_path}")
    except IOError:
        print(f"错误：无法写入 CSV 文件：{output_csv_path}")
    except Exception as e:
        print(f"写入 CSV 文件时发生错误：{e}")


# --- 使用示例 ---
if __name__ == "__main__":
    # ****** 修改这里 ******
    coco_json_file = '../Native/Mask-51.json' # <--- 替换为你的 JSON 文件路径
    output_csv_file = 'output_results.csv'             # <--- 替换为你想要的输出 CSV 文件名
    # *********************

    if not os.path.exists(coco_json_file):
         print(f"错误: 输入的 JSON 文件不存在: {coco_json_file}")
    else:
        process_coco_json(coco_json_file, output_csv_file)
