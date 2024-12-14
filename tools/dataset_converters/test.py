import json
import numpy as np

def rle_encode(matrix):
    """Run-length encoding for a 2D numpy array."""
    flat_matrix = matrix.flatten()
    value = flat_matrix[0]
    count = 0
    rle = []
    for val in flat_matrix:
        if val == value:
            count += 1
        else:
            rle.append((value, count))
            value = val
            count = 1
    rle.append((value, count))
    return rle

def rle_decode(rle, shape):
    """Run-length decoding for a 2D numpy array."""
    flat_matrix = []
    for value, count in rle:
        flat_matrix.extend([value] * count)
    return np.array(flat_matrix).reshape(shape)

def save_rle_to_json(matrix, json_path):
    """Save RLE encoded matrix to a JSON file."""
    rle = rle_encode(matrix)
    data = {
        "shape": matrix.shape,
        "rle": rle
    }
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)

def load_rle_from_json(json_path):
    """Load RLE encoded matrix from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    shape = tuple(data["shape"])
    rle = data["rle"]
    print(rle)
    return rle_decode(rle, shape)

# 示例用法
matrix = np.random.rand(10, 10)
matrix[matrix < 0.8] = 0  # 使其稀疏

# 保存到JSON文件
save_rle_to_json(matrix, "sparse_matrix.json")

# 从JSON文件加载
loaded_matrix = load_rle_from_json("sparse_matrix.json")
print(loaded_matrix)
print(np.array_equal(matrix, loaded_matrix))  # 应输出True