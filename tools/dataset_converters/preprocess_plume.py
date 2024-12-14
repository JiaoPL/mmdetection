import h5py
import os
import random
import string
import scipy
import json
import numpy as np

data_map = {
    "l1_plume": "L1_with_plume_1000",
    "l1_wind": "L1_wind_1000",
    "l1_instance": "L1_instance_1000"
}
bands = 48

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


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


def generate_anninfo(data_path: str, ann_path: str):
    ann_json = {}
    
    l1_plume_path = os.path.join(data_path, data_map["l1_plume"])
    l1_wind_path = os.path.join(data_path, data_map["l1_wind"])
    l1_instance_path = os.path.join(data_path, data_map["l1_instance"])
    l1_plume_files = sorted(os.listdir(l1_plume_path))
    
    for i, l1_plume_filename in enumerate(l1_plume_files):
        if i > 100:
            break
        ann_json[i] = {}

        common_identifier = l1_plume_filename.replace("L1_with_plume_", "")
        l1_wind_filename = f"L1_wind_{common_identifier}"
        l1_instance_filename = f"L1_instance_{common_identifier}"
        
        with h5py.File(os.path.join(l1_plume_path, l1_plume_filename),  "r", libver='latest', swmr=True) as plume_data:
            bcenters = plume_data["bcenter"][:]
            if bcenters.shape[0] != bands:
                continue
            
            bands_data = plume_data["L1_with_plume"][:]
            c, h, w = bands_data.shape

            ann_json[i]["plume_path"] = os.path.join(l1_plume_path, l1_plume_filename)
            ann_json[i]["wind_path"] = os.path.join(l1_wind_path, l1_wind_filename)
            ann_json[i]["img_id"] = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            ann_json[i]["height"] = h
            ann_json[i]["width"] = w
            ann_json[i]["channel"] = c
        
        ann_json[i]["instances"] = []
        instance_data = scipy.io.loadmat(os.path.join(l1_instance_path, l1_instance_filename))
        for j in range(len(instance_data['instance'][0][0])):
            bbox = instance_data['instance'][0][0][j][0][0][2][0].tolist()
            mask = instance_data['instance'][0][0][j][0][0][1]
            rle_mask = rle_encode(mask)
            bbox_label = 0  # only one class: plume
            ignore_flag = False
            emission = instance_data['instance'][0][0][j][0][0][3][0][0]
            cur_id = instance_data['instance'][0][0][j][0][0][0][0][0]
            ann_json[i]["instances"].append({
                "bbox": bbox,
                "ignore_flag": ignore_flag,
                "bbox_label": bbox_label,
                "mask": {
                    "size": [h, w],
                    "counts": rle_mask
                },
                "emission": emission,
                "id": cur_id,
            })

    with open(ann_path, "w", encoding="utf-8") as json_file:
        json.dump(ann_json, json_file, default=convert_to_serializable, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    datapath = u"E:\RS\experiments\L1_datasets_1000"
    ann_dir = os.path.join(datapath, "ann")
    if not os.path.exists(ann_dir):
        os.makedirs(ann_dir)
    generate_anninfo(data_path=datapath, ann_path=os.path.join(ann_dir, "ann_rle_plume_test100.json"))
