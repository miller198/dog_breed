
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, random_split, DataLoader
import torch
import torch.nn as nn # layer들을 호출하기 위해서
import numpy as np
import torch.optim as optim # optimization method를 사용하기 위해서
import torch.nn.init as init # weight initialization 해주기 위해서
from tqdm import tqdm

import os
import pandas as pd
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from collections import OrderedDict
from parallel import DataParallelModel,DataParallelCriterion
# %matplotlib inline

from efficientnet_pytorch import EfficientNet

batch_size = 128
num_workers = 2 * torch.cuda.device_count()
print(num_workers)
print(os.cpu_count())

train_set = ImageFolder(root='./stanford/train')
test_set = ImageFolder(root='./stanford/test')

n_class = len(train_set.classes)


dataset_size = len(train_set)
val_pct = 0.2
val_size = int(dataset_size*val_pct)
train_size = dataset_size - val_size
test_size = len(test_set)

train_size, val_size, test_size


train_ds, val_ds = random_split(train_set, [train_size, val_size])
len(train_ds), len(val_ds)

# Image Transform을 지정합니다.
train_transform = transforms.Compose([
#        transforms.Resize((128, 128)),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(256, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),
#    transforms.Normalize(*imagenet_stats, inplace=True)
])

val_transform = transforms.Compose([
#     transforms.Resize((128, 128)),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
#    transforms.Normalize(*imagenet_stats, inplace=True)
])

test_transform = transforms.Compose([
#     transforms.Resize((128, 128)), 
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
#    transforms.Normalize(*imagenet_stats, inplace=True)
])

# 이미지 폴더로부터 데이터를 로드합니다.
class Dataset(Dataset):
    
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)  
            return img, label
        
        
train_dataset = Dataset(train_ds, train_transform)
val_dataset = Dataset(val_ds, val_transform)
test_dataset = Dataset(test_set, test_transform)


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size*2, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=num_workers, pin_memory=True)


# 현재 배치를 이용해 격자 형태의 이미지를 만들어 시각화

# def show_batch(dl):
#     for img, lb in dl:
#         fig, ax = plt.subplots(figsize=(16, 8))
#         ax.set_xticks([]); ax.set_yticks([])
#         ax.imshow(make_grid(img.cpu(), nrow=16).permute(1,2,0))
#         break
        
# show_batch(train_loader)


images, classes = next(iter(train_loader))

images.shape, classes.shape


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ModelBase(nn.Module):
    # training step
    def training_step(self, batch):
        img, targets = batch
        out = self(img)
        loss = F.nll_loss(out, targets)
        return loss
    
    # validation step
    def validation_step(self, batch):
        img, targets = batch
        out = self(img)
        loss = F.nll_loss(out, targets)
        acc = accuracy(out, targets)
        return {'val_acc':acc.detach(), 'val_loss':loss.detach()}
    
    # validation epoch end
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss':epoch_loss.item(), 'val_acc':epoch_acc.item()}
        
    # print result end epoch
    def epoch_end(self, epoch, result):
        print("Epoch [{}] : train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result["train_loss"], result["val_loss"], result["val_acc"]))
        


class PretrainedEfficientNet_V2(ModelBase):
    def __init__(self):
        super().__init__()
        
        self.network = EfficientNet.from_pretrained('efficientnet-b4')
#         Replace last layer
        num_ftrs = self.network._fc.in_features
        self.network._fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_class),
            nn.LogSoftmax(dim=1)
        )
        self.network = nn.DataParallel(self.network,device_ids=[0,1,2,3,4])
#         self.network = DataParallelModel(self.network)
        
    def forward(self, xb):
        return self.network(xb)


model = PretrainedEfficientNet_V2()
model


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(d, device) for d in data]
    else:
        return data.to(device, non_blocking=True)

    
class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)


# getting default device
device = torch.device('cuda')
print(device)

# moving train dataloader and val dataloader to gpu
train_dl = DeviceDataLoader(train_loader, device)
val_dl = DeviceDataLoader(val_loader, device)


# moving model to gpu
to_device(model, device);


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0, grad_clip=None, opt_func = torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # set up one cycle lr scheduler
    #초기 learing rate에서 1cycle annealing하는 scheduler이다. 1주기 전략은 
    #초기 learning rate에서 최대 learning rate까지 올라간 후 초기 learning rate보다 훨씬 낮은 learning rate로 annealing한다. 
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        
        # Training phase
        model.train()       
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            
            # calculates gradients
            loss.backward()
            
            # check gradient clipping 
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            # perform gradient descent and modifies the weights
            optimizer.step()
            
            # reset the gradients
            optimizer.zero_grad()
            
            # record and update lr
            lrs.append(get_lr(optimizer))
            
            # modifies the lr value
            sched.step()
            
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
        
        
    return history
        
    

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


evaluate(model, val_dl) 


opt_func = torch.optim.SGD
num_epoch = 30
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4


history = fit_one_cycle(num_epoch, max_lr, model, train_dl, val_dl, weight_decay, grad_clip, opt_func)


num_epoch = 5
max_lr = 0.001
weight_decay = 1e-5

history += fit_one_cycle(num_epoch, max_lr, model, train_dl, val_dl, weight_decay, grad_clip, opt_func)


num_epoch = 5
max_lr = 0.01

history += fit_one_cycle(num_epoch, max_lr, model, train_dl, val_dl, weight_decay, grad_clip, opt_func)


num_epoch = 5
max_lr = 0.01

history += fit_one_cycle(num_epoch, max_lr, model, train_dl, val_dl, weight_decay, grad_clip, opt_func)


val_loss = []
train_loss = []
val_acc = []
time = list(range(len(history)))
for h in history:
    val_loss.append(h['val_loss'])
    train_loss.append(h['train_loss'])
    val_acc.append(h['val_acc'])


plt.plot(time, val_loss, c='red', label='val_loss', marker='x')
plt.plot(time, train_loss, c='blue', label='train_loss', marker='x')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()


plt.plot(time, val_acc, c='red', label='accuracy', marker='x')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()


lrs = np.concatenate([x.get('lrs', []) for x in history])
plt.xlabel('epochs')
plt.ylabel('lr')
plt.plot(lrs)
plt.show()


def predict_single(img, label):
    xb = img.unsqueeze(0) # adding extra dimension
    xb = to_device(xb, device)
    preds = model(xb)                   # change model object here
    predictions = preds[0]
    
    max_val, kls = torch.max(predictions, dim=0)
    print('Actual :', breeds[label], ' | Predicted :', breeds[kls])
    plt.imshow(img.permute(1,2,0))
    plt.show()


breeds = []

def rename(name):
    return ' '.join(' '.join(name.split('-')[1:]).split('_'))

for n in train_set.classes:
    breeds.append(rename(n))


predict_single(*test_dataset[2427])


test_dl = DeviceDataLoader(test_loader, device)


result = evaluate(model, test_dl)
result


torch.save(model, './models/dataloader-2_EfficV2_Acc90.pt')


torch.save(model.state_dict(), './models/dataloader-2_EfficV2_Acc90.pt')


