from pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

RUN pip install transformers==4.51.3 datasets==3.6.0
RUN pip install flwr==1.18.0

# download model and dataset
RUN python3 -c "from torchvision.models import vit_b_16, ViT_B_16_Weights; vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)"
RUN python3 -c "from datasets import load_dataset; load_dataset('Honaker/eurosat_dataset', split='train')"

WORKDIR /workspace

COPY finetune.py .

CMD ["python3","finetune.py"]
