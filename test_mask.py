import cv2

import h5py
import numpy as np


tag_file = "E:\RS\experiments\L1_datasets_1000\L1_tag_1000\L1_tag_n1_E111.7_N35.3_20220518_rd4.mat"
with h5py.File(tag_file, "r", libver='latest', swmr=True) as tag_data: 
    mask_data = tag_data["L1_tag"][:]


def mask_to_bbox(mask):
    """
    根据mask生成bbox
    :param mask: 二维数组，表示掩码图像
    :return: bbox的坐标列表，每个bbox为 [x_min, y_min, x_max, y_max]
    """
    bboxes = []
    
    # 获取mask中的每个唯一值（每个对象）
    object_ids = np.unique(mask)
    
    # 移除背景 (通常背景为0)
    object_ids = object_ids[object_ids != 0]
    
    for obj_id in object_ids:
        # 创建单个对象的二值掩码
        obj_mask = (mask == obj_id).astype(np.uint8)
        
        # 查找轮廓
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        
        for contour in contours:
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [x, y, x + w, y + h]
            bboxes.append(bbox)
    
    return bboxes


def save_mask_as_image(mask, output_path="mask_image.png"):
    """
    将mask导出为图像文件
    :param mask: numpy 数组，二维，表示掩码
    :param output_path: 输出的图像路径（默认为 mask_image.png）
    """
    # 将不同对象分配不同颜色
    unique_ids = np.unique(mask)
    color_map = {obj_id: np.random.randint(0, 255, size=(3,)).tolist() for obj_id in unique_ids if obj_id != 0}
    
    # 初始化彩色图像
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for obj_id, color in color_map.items():
        color_mask[mask == obj_id] = color  # 将物体 ID 对应的像素值设置为指定颜色

    # 保存图像
    cv2.imwrite(output_path, color_mask)
    print(f"Mask saved as image at {output_path}")
    

save_mask_as_image(mask_data)
bboxes = mask_to_bbox(mask_data)
print(bboxes)

