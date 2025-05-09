### Install dependencies


If finetuning the ViT:
```shell
pip install torch==2.7.0 torchvision==0.22.0
pip install transformers==4.51.3 datasets==3.6.0
pip install flwr==1.18.0
```

If finetuning an LLM with Unsloth:

```shell
pip install torch==2.7.0 torchvision==0.22.0
pip install unsloth==2025.4.7
pip install flwr==1.18.0
```

### Run

Either finetune a ViT or an LLM.

For ViT, ensure you first have a local copy of the dataset and model. This primarily done so the Docker cotaniner built for the ViT doesn't require internet access.

```shell
python -c "from datasets import load_dataset; dd=load_dataset('Honaker/eurosat_dataset', split='train', cache_dir='./eurosat_dataset'); dd.save_to_disk('./eurosat_dataset')"
python -c "import torch; from torchvision.models import vit_b_16, ViT_B_16_Weights; model=vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1); torch.save(model.state_dict(), 'vit.pth')"
```


```shell
python finetune.py
# or the below for an LLM
# python finetune_unsloth.py
```

Expect a log as follows showing the avg training loss every 50 batches:

```shell
Flower finetune losses: [0.035802897065877914, 0.016872126170817542, 0.012768076635934043, 0.01073403720318383, 0.009541337469833629, 0.00869496681438054, 0.008089264186464275]
```

Expect log as follows if you use the `finetune_unsloth.py` script:

```shell
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
INFO :      ClientApp starting finetuning for max_steps = 128
==((====))==  Unsloth 2025.4.7: Fast Llama patching. Transformers: 4.51.3.
   \\   /|    NVIDIA GeForce RTX 3090. Num GPUs = 2. Max memory: 23.588 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.7.0+cu126. CUDA: 8.6. CUDA Toolkit: 12.6. Triton: 3.3.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.30. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Unsloth: Making `model.base_model.model.model` require gradients
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 210,289 | Num Epochs = 1 | Total steps = 128
O^O/ \_/ \    Batch size per device = 16 | Gradient accumulation steps = 4
\        /    Data Parallel GPUs = 1 | Total batch size (16 x 4 x 1) = 64
 "-____-"     Trainable parameters = 11,272,192/1,000,000,000 (1.13% trained)
  0%|                                                  | 0/128 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
Unsloth: Will smartly offload gradients to save VRAM!
{'loss': 2.6495, 'grad_norm': 1.1488209962844849, 'learning_rate': 0.0, 'epoch': 0.0}
{'loss': 2.4049, 'grad_norm': 1.1651126146316528, 'learning_rate': 5e-06, 'epoch': 0.0}
{'loss': 2.5544, 'grad_norm': 1.2086200714111328, 'learning_rate': 1e-05, 'epoch': 0.0}                                      
[...]                
{'loss': 1.6611, 'grad_norm': 0.6909930109977722, 'learning_rate': 1.6949152542372882e-06, 'epoch': 0.04}                    
{'loss': 1.8031, 'grad_norm': 0.7734246850013733, 'learning_rate': 1.2711864406779662e-06, 'epoch': 0.04}                    
{'loss': 1.6882, 'grad_norm': 0.7623358368873596, 'learning_rate': 8.474576271186441e-07, 'epoch': 0.04}                     
{'loss': 1.745, 'grad_norm': 0.7319967746734619, 'learning_rate': 4.2372881355932204e-07, 'epoch': 0.04}                     
{'train_runtime': 214.2777, 'train_samples_per_second': 38.231, 'train_steps_per_second': 0.597, 'train_loss': 1.8917758977040648, 'epoch': 0.04}
100%|██████████████████████████████████████████████████████████████████████████████████████| 128/128 [03:34<00:00,  1.67s/it]
Final training loss: 1.8917758977040648
INFO :      Final train loss: 1.8917758977040648
```


### With Docker

```shell
docker build -t flwr-spc .
# use instead unsloth.dockerfile for the Unsloth LLM finetuning setup
```

And then run (note you need your host machine to have [Nvidia docker runtime](https://developer.nvidia.com/container-runtime)). By default the ViT finetuning will be exectued:

```shell
docker run --shm-size=1g -it --rm --gpus all flwr-spc
```

Unsloth LLM finetuning will work ok until the point when the finteuning starts, throwing error:

```shell
[...]
  File "/usr/local/lib/python3.11/dist-packages/torch/_inductor/async_compile.py", line 447, in _wait_futures
    raise RuntimeError(
torch._inductor.exc.InductorError: RuntimeError: A compilation subprocess exited unexpectedly. This is likely due to a crash. To facilitate debugging, you can re-run with TORCHINDUCTOR_COMPILE_THREADS=1 to cause compilation to occur in the main process.

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
```
