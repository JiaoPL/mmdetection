import numpy as np
from pycocotools import mask as coco_mask

def custom_rle_to_compressed_rle(rle, height, width):
    """
    将自定义 RLE 转换为 COCO 格式的压缩 RLE。
    """
    # 解码自定义 RLE 到二进制掩码
    def rle_to_mask(rle, height, width):
        mask = np.zeros(height * width, dtype=np.uint8)
        pos = 0
        for value, count in rle:
            mask[pos:pos + count] = value
            pos += count
        return mask.reshape((height, width))
    
    # 1. 解码 RLE 到 2D 掩码
    binary_mask = rle_to_mask(rle, height, width)
    
    # 2. 使用 pycocotools 转换为压缩 RLE 格式
    rle_encoded = coco_mask.encode(np.asfortranarray(binary_mask))
    
    # 3. 将 rle_encoded 转换为 COCO 格式（'size', 'counts'）
    compressed_rle = {
        "size": [height, width],
        "counts": rle_encoded["counts"].decode("utf-8")  # 压缩后的 counts 为字符串
    }
    
    return compressed_rle

# 示例 RLE 数据
rle = [
    [0, 13934], [1, 3], [0, 508], [0, 1], [1, 7], [0, 504], [1, 9], [0, 503],
    [1, 11], [0, 499], [1, 13], [0, 499], [1, 12], [0, 500], [1, 12], [0, 499],
    [1, 14], [0, 497], [1, 15], [0, 496], [1, 16], [0, 496], [1, 12], [0, 500],
    [1, 13], [0, 499], [1, 13], [0, 499], [1, 14], [0, 496], [1, 14], [0, 498],
    [1, 14], [0, 495], [1, 1], [0, 2], [1, 15], [0, 495], [1, 17], [0, 496],
    [1, 16], [0, 1], [1, 1], [0, 494], [1, 17], [0, 494], [1, 2], [0, 220562]
]

# 图像尺寸
height = 512
width = 512

# 转换为压缩 RLE
compressed_rle = custom_rle_to_compressed_rle(rle, height, width)

print("Compressed RLE segmentation:")
print(compressed_rle)
