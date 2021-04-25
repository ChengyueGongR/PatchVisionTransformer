# VisionTransformer


# Getting Started 

cd `timm_talk`

`pip install --upgrade .`

cd `../deit`


# Model Zoo

We provide models pretrained on ImageNet 2012. More models will be uploaded.

| name | acc@1 | acc@5 | #params | url |
| --- | --- | --- | --- | --- |
| VIT-B12 | 82.9 | 96.3 | 86M | [model](https://drive.google.com/file/d/1NEx-fY6q3UvphJItqABCr2DRjzcReCeO/view?usp=sharing) |
| VIT-B24 | 83.3 | 96.4 | 172M| [model](https://drive.google.com/file/d/1TKG7UIQvFTpoMMLffwYEhYDPoCyzXDhu/view?usp=sharing) |
| VIT-B12-384 | 84.2 | 97.0 | 86M | [model](https://drive.google.com/file/d/1ps-DDxjtbS9fdbSspl-LKScs_IZENKaG/view?usp=sharing) |

# Evaluate 

Example: 

Run `python -m torch.distributed.launch --nproc_per_node=XX --master_port=XX --use_env main.py --model deit_base_patch16_224 --aa rand-m9-mstd0.5-inc1  --input-size 224 --batch-size 16 --num_workers 2 --data-path path --output_dir output_dir --resume model.pth --eval`, we will get `Acc@1 82.928 Acc@5 96.342 loss 0.721 `.


