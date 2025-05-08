### Install dependencies

```shell
pip install torch==2.7.0 torchvision==0.22.0
pip install unsloth==2025.4.7
pip install flwr==1.18.0
```

### Run

```shell
python finetune.py
```

Expect log as follows:

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
```

And then run (note you need your host machine to have [Nvidia docker runtime](https://developer.nvidia.com/container-runtime)):

```shell
docker run -it --rm --gpus all flwr-spc
```