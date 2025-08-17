# Active Region segmentation

## How to run 
```sh
uv run torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py
```

## Dataset info 
- Input : SDO and AIA images of shape (13,4096,4096)
- Output : Masks of shape (4096,4096)

