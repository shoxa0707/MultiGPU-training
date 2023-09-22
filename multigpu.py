import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import os

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12346'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
class Trainer:
    def __init__(self, model: torch.nn.Module, train_data: DataLoader, optimizer: torch.optim.Optimizer, gpu_id: int, save_every: int):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs = 0
        
        # if os.path.exists(checkpoint_path):
        #     print("loading checkpoint...")
        #     self.load_checkpoint(checkpoint_path)
        
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.epochs = checkpoint['epochs']
        
    def run_batch(self, source, targets):
        self.optimizer.zero_grad()
        outputs = self.model(source)
        loss = torch.nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        self.optimizer.step()
    
    def run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] | EPOCH {epoch} | BatchSize: {batch_size} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self.run_batch(source, targets)
    
    def save_checkpoint(self, epoch):
        checkpoint = {}
        checkpoint['model_state'] = self.model.module.state_dict()
        checkpoint['epoch'] = epoch
        torch.save(checkpoint, 'checkpoint.pt')
        print("Checkpoint saved checkpoint.pt")
        
    def train(self, max_epochs):
        for epoch in range(max_epochs):
            self.run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self.save_checkpoint(epoch)
            elif epoch == max_epochs - 1:
                self.save_checkpoint(epoch)

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
                
                
def load_train():
    train_set = datasets.MNIST(
        root="data",
        train=True,
        download=False,
        transform=ToTensor()
    )
    model = MNIST()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    return train_set, model, optimizer

def prepare_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(dataset))
    
def main(rank, world_size, total_epochs, save_every):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train()
    train_data = prepare_dataloader(dataset, batch_size=32)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == '__main__':
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, total_epochs, save_every), nprocs=world_size)
    
