import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
import torch.nn.functional as F

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed  as dist
import os
from datetime import timedelta

# server_store = dist.TCPStore("192.169.0.128", 1235, 3, True, timedelta(seconds=30))
print("connection waiting...")
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
print("success")

class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(16, 64, 3, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding='same')
        self.bn4 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(3136, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn5(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
def run_epoch(epoch):
    b_sz = len(next(iter(train_data))[0])
    print(f"[GPU{rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(train_data)}")
    for source, targets in train_data:
        source = source.to(device_id)
        targets = targets.to(device_id)
        run_batch(source, targets)

def run_batch(source, targets):
    optimizer.zero_grad()
    output = model(source)
    loss = torch.nn.NLLLoss()(output, targets)
    loss.backward()
    optimizer.step()

def save_checkpoint(epoch):
    check = {}
    check["MODEL_STATE"] = model.module.state_dict()
    check["EPOCHS_RUN"] = epoch
    torch.save(check, f"checkpoint{epoch}.pt")
    print(f"Epoch {epoch} | Training checkpoint saved at checkpoint{epoch}.pt")

def train(epochs):
    for epoch in range(epochs):
        run_epoch(epoch)
        # if epoch % 10 == 0:
        #     save_checkpoint(epoch)
    
                
def load_train_objs():
    train_set = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    model = MNIST()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )    


import sys
total_epochs = int(sys.argv[1])

dataset, model, optimizer = load_train_objs()
device_id = rank % torch.cuda.device_count()
# device_id = torch.device("cuda:1")
print(device_id)
model = model.to(device_id)  # equivalent to `output_device` in DDP
model = DDP(model, device_ids=[device_id])

train_data = prepare_dataloader(dataset, batch_size=32)

train(total_epochs)
dist.destroy_process_group()
