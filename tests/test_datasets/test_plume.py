# Copyright (c) OpenMMLab. All rights reserved.

from mmdet.datasets import CH4PlumeDataset
from mmdet.datasets.transforms import LoadPlumeMat, FourioerTransform, LoadPlumeAnnotations



def test_plume_dataset():
    pipelines = []
    load_plume = LoadPlumeMat()
    load_ann = LoadPlumeAnnotations(with_bbox=True, with_mask=True, with_label=True)
    fourier_transform = FourioerTransform()
    pipelines.append(load_plume)
    pipelines.append(load_ann)
    pipelines.append(fourier_transform)
    # test CH4PlumeDataset
    dataset = CH4PlumeDataset(
        ann_file=u"E:\\RS\\experiments\\L1_datasets_1000\\ann\\ann_rle_plume.json",
        pipeline=pipelines,
        serialize_data=False,
        lazy_init=False)
    breakpoint()



if __name__ == '__main__':
    test_plume_dataset()