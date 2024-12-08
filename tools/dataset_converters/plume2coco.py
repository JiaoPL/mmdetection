import json
import os
from datetime import datetime
import numpy as np
from pycocotools import mask as coco_mask

# 获取当前日期
today_date = datetime.now().strftime("%Y-%m-%d")


# 初始化 COCO 格式的结构
coco_format = {
    "info": {
        "description": "Methane Plume Dataset",
        "version": "1.0",
        "year": int(today_date.split("-")[0]),
        "contributor": "Penglong Jiao",
        "date_created": today_date
    },
    "licenses": [
        {
            "id": 1,
            "name": "Custom License",
            "url": "http://custom-license.org"
        }
    ],
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 0,
            "name": "methane plume",
            "supercategory": "plume"
        }
    ]
}


annid = 0
imgid = 0


def rle_decode(rle, img_h, img_w):
    """Run-length decoding for a 2D numpy array."""
    flat_matrix = []
    try:
        for value, count in rle:
            flat_matrix.extend([value] * count)
    except:
        print(f"rle: {rle}" , flush=True)
    return np.array(flat_matrix, dtype=np.uint8).reshape((img_h, img_w))

def custom_to_coco(custom_ann):
    global annid, imgid, coco_format
    # 解析 custom ann 的内容
    image_data = {
        "id": imgid,
        "width": custom_ann["width"],
        "height": custom_ann["height"],
        "date_captured": today_date,
        "channel": custom_ann["channel"],
        "plume_path": custom_ann["plume_path"],
        "wind_path": custom_ann["wind_path"]
    }
    coco_format["images"].append(image_data)

    # 转换 annotations
    for instance in custom_ann["instances"]:
        rle = instance["mask"]["counts"]
        size = instance["mask"]["size"]
        mask = rle_decode(rle, custom_ann["height"], custom_ann["width"])
        
        # 计算 area      
        area = int(mask.sum())
        
        rle_encoded = coco_mask.encode(np.asfortranarray(mask))
        compressed_rle = {
            "size": size,
            "counts": rle_encoded["counts"].decode("utf-8")  # 压缩后的 counts 为字符串
        }

        annotation = {
            "id": annid,
            "image_id": imgid,
            "category_id": 0,  # 固定为 "plume"
            "bbox": instance["bbox"],
            "segmentation": compressed_rle,  # 保持 segmentation 的 RLE 编码
            "area": area,
            "iscrowd": 0,  # 固定为 0
            "emission": instance["emission"]  # 自定义字段
        }
        annid += 1
        coco_format["annotations"].append(annotation)

    imgid += 1



plume_annpath = r"E:\RS\experiments\L1_datasets_1000\ann\ann_rle_plume.json"
new_plume_annname = "plume_ann_coco.json"
new_plume_annpath = os.path.join(os.path.dirname(plume_annpath), new_plume_annname)


# 读取 custom 格式的 annotation
with open(plume_annpath, "r") as f:
    custom_anns = json.load(f)

# 转换为 COCO 格式
count = 0
for _, custom_ann in custom_anns.items():
    if custom_ann:
        count += 1
        custom_to_coco(custom_ann)


# 保存为 JSON 文件
with open(new_plume_annpath, "w") as f:
    json.dump(coco_format, f, indent=2)

print(f"Total {count} images processed.")
print(f"Conversion completed. Saved to {new_plume_annname}")
