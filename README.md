# VisionTransformer


Changes:
1. in `timm_talk/timm/models/vision_transformer.py`, an additional dropout is added to the classification head.
2. in `timm_talk/timm/models/vision_transformer.py`, talking head attention is added to the attention layer.
3. in `timm_talk/timm/utils/cuda.py`, the default `clip norm` is set to 5.
4. in `deit/main.py`, the default min_lr is set to be `1e-7'.
5. drop path, ema, and augmentation is changed.

cd `timm_talk`

`pip install --upgrade .`

cd `../deit`

`python -m torch.distributed.launch --nproc_per_node=8 --master_port=1511 --use_env main.py --model deit_small_patch16_224 --aa rand-m9-mstd0.5-inc1  --input-size 224 --batch-size 128 --num_workers 5 --data-path /scratch/cluster/dilin/datasets/imagenet --no-repeated-aug --output_dir /scratch/cluster/cygong/vision_transformer_small_talk --smoothing 1e-1 --weight-decay 5e-2 --lr 5e-4 --epochs 360 --model-ema-decay 0.99999 --drop-path 0.2 --drop 0.0 --mixup 0.0 --mixup-switch-prob 0.0`

`python -m torch.distributed.launch --nproc_per_node=8 --master_port=1511 --use_env main.py --model deit_base_patch16_224 --aa rand-m9-mstd0.5-inc1 --input-size 224 --batch-size 72 --num_workers 2 --data-path /scratch/cluster/dilin/datasets/imagenet --no-repeated-aug --output_dir /scratch/cluster/cygong/vision_transformer_base_talk --smoothing 1e-1 --weight-decay 5e-2 --lr 5e-4 --epochs 360 --model-ema-decay 0.99999 --drop-path 0.5 --drop .0 --mixup .0 --mixup-switch-prob 0.0`
