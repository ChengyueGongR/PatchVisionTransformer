nvidia-smi
top -bn 1 -c -i

bash
source activate pytorch
# nvidia-smi
export OMP_NUM_THREADS=4
cd ../timm
pip install --upgrade .
cd ../deit

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_small_patch16_224 --batch-size 256 --data-path /scratch/cluster/dilin/datasets/imagenet  --output_dir /scratch/cluster/cygong/vision_transformer_fixup_small
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1512 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path /scratch/cluster/dilin/datasets/imagenet  --output_dir /scratch/cluster/cygong/vision_transformer_shuffle # --resume /scratch/cluster/cygong/vision_transformer_best/checkpoint.pth

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1510 --use_env main.py --model deit_small_patch16_224 --batch-size 120 --data-path /scratch/cluster/dilin/datasets/imagenet  --output_dir /scratch/cluster/cygong/vision_transformer_small_unshare_multimlp_coefficient_efficient




