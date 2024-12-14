import copy
import os.path as osp
import json
from typing import Callable, List, Mapping, Optional, Sequence
from typing import Any
from mmengine.config import Config
from mmengine.dataset import BaseDataset
from mmengine.fileio import load, get_local_path
from mmengine.utils import is_abs

from mmdet.registry import DATASETS


@DATASETS.register_module()
class CH4PlumeDataset(BaseDataset):
    METAINFO = {
        'classes': ('CH4_Plume'),
        'palette': [(255, 241, 0)],
    }
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        with open(self.ann_file, 'r', encoding='utf-8') as f:
            ann_json = json.load(f)
        data_list = []
        for _, ann in ann_json.items():
            data_list.append(ann)

        return data_list



    