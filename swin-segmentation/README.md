# Segmentation

This repository contains PyTorch evaluation code, training code and pretrained models for segmentation. It is based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.11.0) and [Swin-Transformer-Semantic-Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)

We will further upload results on other model architectures.

# Getting Started 

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md#installation) for installation and dataset preparation.

# Model Zoo

We provide models pretrained on ImageNet 2012 and finetune the checkpoint on segmentation datasets.

| name | acc@1 | #params | url |
| --- | --- | --- | --- |
| SWIN-Large | 87.4 | 197M | [model](https://drive.google.com/file/d/1elIVsE_W5jHCfBSALCjF0f79Unk-bmS0/view?usp=sharing) |

We finetune the checkpoint and get following results.

| name | mIoU | mIoU (ms + flip) | #params | url |
| --- | --- | --- | --- | --- |
| ADE20K | 83.0 | 54.4 | 234M | [model](https://drive.google.com/file/d/1InHGA0cqDUQwi1qZCZtXiFet9uHyw91c/view?usp=sharing), [log](https://drive.google.com/file/d/1va6Ptawr5C7bhGchz-028wrzObUgseHE/view?usp=sharing)|
| CityScapes | 82.9 | 83.9 | 234M | [model](https://drive.google.com/file/d/1z69_V6JPqq5oR7aJOgw9YVyNjktEX3sV/view?usp=sharing), [log](https://drive.google.com/file/d/1j0Hub-HeMCUbeHhnGw79FHtvvxvSCz1y/view?usp=sharing)|

# Train


For ADE20K, run: 
```
bash tools/dist_train.sh configs/swin/upernet_swin_large_patch4_window12_512x512_160k_ade20k.py 8 --options model.pretrained=ckpt.pth model.backbone.use_checkpoint=True data.workers_per_gpu=1 work_dir=work_dir
```
From the checkpoint ([model](https://drive.google.com/file/d/1SNRD-pHQ8LBW96oHcOUO6A652Tw1z-Hu/view?usp=sharing), [log](https://drive.google.com/file/d/1-XL912aMs5E-rRfZeqknKZN30qsgO17B/view?usp=sharing)), further change iterations to 200K, drop_path of the backbone to 0.5, run:
```
bash tools/dist_train.sh config 8 --options model.pretrained=ckpt.pth model.backbone.use_checkpoint=True data.workers_per_gpu=1 work_dir=work_dir --resume-from work_dir/exp_name/latest.pth
```
Finally evaluate with
```
bash tools/dist_test.sh configs/swin/upernet_swin_large_patch4_window12_512x512_160k_ade20k.py work_dir/exp_name/latest.pth 2 --aug-test --eval mIoU
```

For Cityscapes, run: 
```
bash tools/dist_train.sh configs/swin/upernet_swin_large_patch4_window12_1025x1025_160k_cityscapes.py 8 --options model.pretrained=ckpt.pth model.backbone.use_checkpoint=True data.workers_per_gpu=1 work_dir=work_dir
```
Finally evaluate with
```
bash tools/dist_test.sh configs/swin/upernet_swin_large_patch4_window12_1025x1025_160k_cityscapes.py work_dir/exp_name/latest.pth 2 --aug-test --eval mIoU
```
