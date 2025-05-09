from pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN pip install transformers==4.51.3 datasets==3.6.0
RUN pip install flwr==1.18.0

WORKDIR /workspace

# Download model and dataset
RUN python3 -c "import torch; from torchvision.models import vit_b_16, ViT_B_16_Weights; model=vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1); torch.save(model.state_dict(), 'vit.pth')"
RUN python3 -c "from datasets import load_dataset; dd=load_dataset('Honaker/eurosat_dataset', split='train', cache_dir='./eurosat_dataset'); dd.save_to_disk('./eurosat_dataset')"

COPY finetune.py .

CMD ["python3","finetune.py"]
