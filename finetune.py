
from logging import INFO

from datasets import load_dataset

from flwr.common.logger import log
from flwr.common import Context, Message, RecordDict, MetricRecord, ConfigRecord
from flwr.client import ClientApp


import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomResizedCrop,
)

def apply_train_transforms(batch):
    """Apply a very standard set of image transforms."""
    transforms = Compose(
        [
            RandomResizedCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch

def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    avg_loss = 0
    count = 0
    for _ in range(epochs):
        for b_id, batch in enumerate(trainloader):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            avg_loss += loss.item() / labels.shape[0]
            loss.backward()
            optimizer.step()
            count += 1
            if b_id % 50 == 0:
                log(INFO, f"Loss @ step {count}: {avg_loss/count}")

    return avg_loss / len(trainloader)

def task(max_steps: int) -> float:

    # Instantiate a pre-trained ViT-B on ImageNet
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    # We're going to federated the finetuning of this model
    # let's freeze everything except the head
    in_features = model.heads[-1].in_features
    model.heads[-1] = torch.nn.Linear(in_features, 10)

    # Disable gradients for everything
    model.requires_grad_(False)
    # Now enable just for output head
    model.heads.requires_grad_(True)


    train_data = load_dataset("Honaker/eurosat_dataset", split="train")

    trainset = train_data.with_transform(apply_train_transforms)

    trainloader = DataLoader(
        trainset, batch_size=64, num_workers=2, shuffle=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    avg_train_loss = train(
        model, trainloader, optimizer, epochs=1, device=device
    )

    return avg_train_loss


# Flower ClientApp
app = ClientApp()

@app.train("finetune")
def finetune(msg: Message, context: Context):

    max_steps = msg.content["config"]["max-steps"]
    
    log(INFO, f"ClientApp starting finetuning for {max_steps = }")

    final_train_loss = task(max_steps)

    reply_content = RecordDict({"results": MetricRecord({"train-loss": final_train_loss})})

    return Message(content=reply_content, reply_to=msg)



if __name__ == "__main__":

    # Construct a Message
    max_steps = 128
    msg = Message(content=RecordDict({"config": ConfigRecord({'max-steps': max_steps})}), dst_node_id=123, message_type="train.finetune")

    # Process Message with ClientApp
    reply_message = app(message=msg, context=Context)
    final_loss = reply_message.content["results"]["train-loss"]
    log(INFO, f"Final finetuning loss: {final_loss}")