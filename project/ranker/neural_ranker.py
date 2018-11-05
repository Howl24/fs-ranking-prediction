import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

class RankerDataset(Dataset):
    def __init__(self, metafeatures, featsel_methods, target):
        self.metafeatures = metafeatures.astype(np.float32)
        self.featsel_methods = featsel_methods.astype(np.int64)
        self.target = target.astype(np.float32) \
                        if target is not None else \
                        np.zeros(len(metafeatures)).astype(np.float32)
    def __len__(self):
        return len(self.metafeatures)
    
    def __getitem__(self, idx):
        return [self.metafeatures[idx], self.featsel_methods[idx], \
                self.target[idx]]
    
class RankNet(nn.Module):
    def __init__(self, metafeatures_sz, featsel_methods_sz, latent_sz):
        super().__init__()
        self.metafeatures_sz = metafeatures_sz
        self.featsel_methods_sz = featsel_methods_sz
        
        self.linear = nn.Linear(metafeatures_sz, latent_sz)
        self.embedding = nn.Embedding(featsel_methods_sz, latent_sz)
        self.emb_init(self.embedding)
        nn.init.kaiming_normal_(self.linear.weight.data)
        
    def forward(self, metafeatures, featsel_method):
        latent_metafeatures = self.linear(metafeatures)
        latent_featsel_method  = self.embedding(featsel_method)
        output = (latent_metafeatures * latent_featsel_method).sum(1)
        # return output
        return torch.sigmoid(output) # * (self.featsel_methods_sz - 1) + 1
    
    def emb_init(self, x):
        x = x.weight.data
        sc = 2 / (x.size(1) + 1)
        x.uniform_(-sc, sc)
    
def train_step(model, dataset, ranker, target, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    preds = model(dataset, ranker)
    loss = criterion(preds.view(-1), target)
    loss.backward()
    optimizer.step()
    return loss.item()

def get_loss(preds, targets):
    return nn.MSELoss()(torch.FloatTensor(preds), \
                        torch.FloatTensor(targets)).item()

def train_model(model, train_loader, optimizer, criterion,
                n_epochs, print_every=1, USE_CUDA=False, val_loader=None):
    train_losses = []
    val_losses = []
    for epoch in range(n_epochs):
        train_loss = 0
        for batch_idx, (dataset, ranker, target) in enumerate(train_loader):
            train_loss += train_step(model, dataset, ranker, target, 
                                     optimizer, criterion)
            if batch_idx > 0 and print_every > 0 and \
               batch_idx % print_every == 0:
                train_loss /= print_every
                print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_idx * len(dataset), 
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), train_loss),
                      end='')
                train_losses.append(train_loss)
                train_loss = 0
        
        if val_loader is not None and print_every > 0:
            targets, preds = get_predictions(model, val_loader)
            val_loss = get_loss(preds, targets)
            val_losses.append(val_loss)
            print('\tVal Loss: {:.6f}'.format(val_loss), end='')
        
        if print_every > 0:
            print()
    return train_losses, val_losses

def get_predictions(model, data_loader, USE_CUDA=False):
    targets, preds = [], []
    model.eval()
    for batch_idx, (dataset, ranker, target) in enumerate(data_loader):
        with torch.no_grad():
            pred = model(dataset, ranker)
            targets.extend(target.cpu())
            preds.extend(pred.cpu())
            assert len(targets) == len(preds)
    return [x.item() for x in targets], [x.item() for x in preds]
