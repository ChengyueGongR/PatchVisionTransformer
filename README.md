# VisionTransformer

This repository contains PyTorch evaluation code, training code and pretrained models.

We will further upload results on other model architectures.

# Getting Started 

cd `./deit`

Before using it, make sure you have the pytorch-image-models [`timm`] package by [Ross Wightman](https://github.com/rwightman) installed. Note that our work relies of the augmentations proposed in this library. 

# SWIN
We provide models pretrained on ImageNet 2012 and finetune the checkpoint on segmentation datasets.

| name | acc@1 | #params | url |
| --- | --- | --- | --- |
| SWIN-Large | 87.4 | 197M | [model](https://drive.google.com/file/d/1elIVsE_W5jHCfBSALCjF0f79Unk-bmS0/view?usp=sharing) |

We finetune the checkpoint and get following results.

| name | mIoU | mIoU (ms + flip) | #params | url |
| --- | --- | --- | --- | --- |
| ADE20K | 83.0 | 54.4 | 234M | [model](https://drive.google.com/file/d/1InHGA0cqDUQwi1qZCZtXiFet9uHyw91c/view?usp=sharing), [log](https://drive.google.com/file/d/1va6Ptawr5C7bhGchz-028wrzObUgseHE/view?usp=sharing)|
| CityScapes | 82.9 | 83.9 | 234M | [model](https://drive.google.com/file/d/1z69_V6JPqq5oR7aJOgw9YVyNjktEX3sV/view?usp=sharing), [log](https://drive.google.com/file/d/1j0Hub-HeMCUbeHhnGw79FHtvvxvSCz1y/view?usp=sharing)|

For more details, please refer to [README](https://github.com/ChengyueGongR/PatchVisionTransformer/blob/main/swin-segmentation/README.md).

# DEIT

## Model Zoo

We provide models pretrained on ImageNet 2012. More models will be uploaded.

| name | acc@1 | acc@5 | #params | url |
| --- | --- | --- | --- | --- |
| VIT-B12 | 82.9 | 96.3 | 86M | [model](https://drive.google.com/file/d/1NEx-fY6q3UvphJItqABCr2DRjzcReCeO/view?usp=sharing) |
| VIT-B24 | 83.3 | 96.4 | 172M| [model](https://drive.google.com/file/d/1TKG7UIQvFTpoMMLffwYEhYDPoCyzXDhu/view?usp=sharing) |
| VIT-B12-384 | 84.2 | 97.0 | 86M | [model](https://drive.google.com/file/d/1ps-DDxjtbS9fdbSspl-LKScs_IZENKaG/view?usp=sharing) |

We finetune the checkpoint proposed by [VIT](https://github.com/google-research/vision_transformer).

| name | acc@1 | acc@5 | #params | url |
| --- | --- | --- | --- | --- |
| VIT-L24 | 83.9 | 96.7 | 305M | [model](https://drive.google.com/file/d/1ByhRxBdb7qp2XF2voHgE3_zJ6mL_0VJW/view?usp=sharing) |
| VIT-L24-384 | 85.4 | 96.7 | 305M | [model](https://drive.google.com/file/d/1ePXsAIzg5HOcd0nolpBTQHh2k2YDa7CM/view?usp=sharing) |

## Evaluate 

For Deit-B12, run: 
```
python -m torch.distributed.launch --nproc_per_node=XX --master_port=XX --use_env main.py --model deit_base_patch16_224 --aa rand-m9-mstd0.5-inc1  --input-size 224 --batch-size 16 --num_workers 2 --data-path path --output_dir output_dir --resume model.pth --eval
```
giving 
```
Acc@1 82.928 Acc@5 96.342 loss 0.721
```

## Train

***The training code is not fully available and the results are currently not reproducable. Please wait for our updates.***

For Deit-B12, run: 
```
python -m torch.distributed.launch --nproc_per_node=XX --master_port=XX --use_env main.py --model deit_base_patch16_224 --aa rand-m9-mstd0.5-inc1  --input-size 224 --batch-size 72 --num_workers 4 --data-path path --output_dir output_dir -no-repeated-aug --epochs 300 --model-ema-decay 0.99996 --drop-path 0.5 --drop .0 --mixup .0 --mixup-switch-prob 0.0 
```
and further refine the model by 
```
python -m torch.distributed.launch --nproc_per_node=XX --master_port=XX --use_env main.py --model deit_base_patch16_224 --aa rand-m9-mstd0.5-inc1  --input-size 224 --batch-size 72 --num_workers 4 --data-path path --output_dir output_dir -no-repeated-aug --start_epoch 300 --epochs 400 --resume model.pth --model-ema-decay 0.99996 --drop-path 0.75 --drop .0 --mixup .0 --mixup-switch-prob 0.0 
```
## Finetune Models Trained on ImageNet-22k

```
python -m torch.distributed.launch --nproc_per_node=XX --master_port=XX --use_env main.py --model deit_large_patch16_224 --aa rand-n1-m1-mstd0.5-inc1 --input-size 224 --batch-size 16 --num_workers 1 --data-path path --output_dir output_dir -no-repeated-aug --smoothing 1e-6 --weight-decay 1e-8 --lr 5e-5 --start_epoch 0 --reprob 1e-6 --resume vit_checkpoint --epochs 40 --model-ema-decay 0.99996 --drop-path 0. --drop .0 --mixup .0 --mixup-switch-prob 0.0 --no-use-talk
```
evaluate 
```
python -m torch.distributed.launch --nproc_per_node=XX --master_port=XX --use_env main.py --model deit_large_patch16_224 --aa rand-n1-m1-mstd0.5-inc1 --input-size 224 --batch-size 16 --num_workers 1 --data-path path --output_dir output_dir -no-repeated-aug --smoothing 1e-6 --weight-decay 1e-8 --lr 5e-5 --start_epoch 0 --reprob 1e-6 --resume vit_checkpoint --epochs 40 --model-ema-decay 0.99996 --drop-path 0. --drop .0 --mixup .0 --mixup-switch-prob 0.0 --no-use-talk --eval 
```
