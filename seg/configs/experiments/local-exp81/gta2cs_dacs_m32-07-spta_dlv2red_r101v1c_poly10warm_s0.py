_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/deeplabv2red_r50-d8.py',
    '../../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    '../../_base_/uda/dacs.py',
    '../../_base_/schedules/adamw.py',
    '../../_base_/schedules/poly10warm.py'
]
gpu_model = 'NVIDIAGeForceRTX4060Ti'
n_gpus = 1
seed = 0
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict())
data = dict(samples_per_gpu=2, workers_per_gpu=1, train=dict())
uda = dict(
    mask_mode='separatetrgaug',
    mask_alpha='same',
    mask_pseudo_threshold='same',
    mask_lambda=1,
    mask_generator=dict(
        type='block', mask_ratio=0.7, mask_block_size=32, _delete_=True))
optimizer_config = None
optimizer = dict(
    lr=6e-05, paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0))))
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')
name = 'gta2cs_dacs_m32-07-spta_dlv2red_r101v1c_poly10warm_s0'
exp = 81
name_dataset = 'gta2cityscapes'
name_architecture = 'dlv2red_r101v1c'
name_encoder = 'r101v1c'
name_decoder = 'dlv2red'
name_uda = 'dacs_m32-0.7-spta'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
