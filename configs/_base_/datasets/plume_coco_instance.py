# dataset settings
dataset_type = 'PlumeDataset'
data_root = u'E:\RS\experiments\L1_datasets_1000\\'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadPlumeMat'),
    dict(type='LoadAnnotationsWithEmission', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='FourioerTransform', cutoff_frequency=0.6, use_spectrum=False),
    dict(type='LogNormalize'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackPlumeInputs')
]
test_pipeline = [
    dict(type='LoadPlumeMat'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotationsWithEmission', with_bbox=True, with_mask=True),
    dict(type='FourioerTransform', cutoff_frequency=0.6, use_spectrum=False),
    dict(type='LogNormalize'),
    dict(type='PackPlumeInputs')
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=u'ann\plume_ann_coco.json',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=u'ann\plume_ann_coco.json',
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader


# TODO: 构造计算回归的metric
val_evaluator = dict(
    type='PlumeCocoMetric',
    ann_file=data_root + u'ann\plume_ann_coco.json',
    metric=['bbox', 'segm'],
    format_only=False,
    outfile_prefix="./work_dirs/plume_val/plume_coco",
    backend_args=backend_args)
test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric=['bbox', 'segm'],
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_instance/test')
