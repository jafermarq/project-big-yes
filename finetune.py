import numpy as np
import torch
from datasets import load_dataset
from flwr.client import ClientApp
from flwr.common import (Array, ArrayRecord, ConfigRecord, Context, Message,
                         RecordDict)
from torch.utils.data import DataLoader
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.transforms import (Compose, Normalize, RandomResizedCrop,
                                    ToTensor)


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


def train(net, trainloader, optimizer, epochs, device) -> list[float]:
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    avg_loss = 0
    count = 0
    losses = []
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
                losses.append(avg_loss / count)

    return losses


def task(local_epochs: int) -> list[float]:

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

    trainloader = DataLoader(trainset, batch_size=64, num_workers=2, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = train(model, trainloader, optimizer, epochs=local_epochs, device=device)

    return losses


# Flower ClientApp
app = ClientApp()


@app.train("finetune")
def finetune(msg: Message, context: Context):

    local_epochs = msg.content["config"]["local-epochs"]

    losses = task(local_epochs)

    reply_content = RecordDict(
        {"results": ArrayRecord({"train-losses": Array(np.array(losses))})}
    )

    return Message(content=reply_content, reply_to=msg)


if __name__ == "__main__":

    # Construct a Message
    local_epochs = 1
    msg = Message(
        content=RecordDict({"config": ConfigRecord({"local-epochs": local_epochs})}),
        dst_node_id=123,
        message_type="train.finetune",
    )

    # Process Message with ClientApp
    reply_message = app(message=msg, context=Context)
    final_loss_array = reply_message.content["results"]["train-losses"].numpy()
    print(f"Flower finetune losses: {final_loss_array.tolist()}")
