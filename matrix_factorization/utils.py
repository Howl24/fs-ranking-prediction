############ DATASET #############

import numpy as np

def get_dataset(dataset, ranking):
    dataset = dataset.repeat(3, axis=0)
    ranker  = np.concatenate([np.zeros(len(ranking)),
                          np.ones(len(ranking)),
                          np.ones(len(ranking)) + 1]).astype(int)
    target  = np.concatenate([ranking.minmax_chi_square_naiveBayes.values,
                              ranking.minmax_fisher_naiveBayes,
                              ranking.minmax_reliefF_naiveBayes])
    return dataset, ranker, target

############ NEURAL NET #############

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

class RankerDataset(Dataset):
    def __init__(self, dataset, ranker, target):
        self.dataset = dataset.astype(np.float32)
        self.ranker = ranker.astype(np.int64)
        self.target = target.astype(np.float32) \
                        if target is not None else \
                        np.zeros(len(dataset)).astype(np.float32)
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return [self.dataset[idx], self.ranker[idx], self.target[idx]]

class RankerNet(nn.Module):
    def __init__(self, dataset_sz, ranker_sz, latent_sz):
        super().__init__()
        self.linear = nn.Linear(dataset_sz, latent_sz)
        self.embedding = nn.Embedding(ranker_sz, latent_sz)
        self.emb_init(self.embedding)
        nn.init.kaiming_normal_(self.linear.weight.data)
        
    # TODO: features => metafeatures, ranker => feature selection
    def forward(self, dataset_features, ranker_index):
        latent_dataset = self.linear(dataset_features)
        latent_ranker  = self.embedding(ranker_index)
        output = (latent_dataset * latent_ranker).sum(1)
        # return output
        return F.sigmoid(output) * 2 + 1
    
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
    return nn.MSELoss()(torch.FloatTensor(preds), torch.FloatTensor(targets)).item()

def train_model(model, train_loader, optimizer, criterion,
                n_epochs, print_every=1, USE_CUDA=False, val_loader=None):
    train_losses = []
    val_losses = []
    for epoch in range(n_epochs):
        train_loss = 0
        for batch_idx, (dataset, ranker, target) in enumerate(train_loader):
            train_loss += train_step(model, dataset, ranker, target, 
                                     optimizer, criterion)
            if batch_idx > 0 and batch_idx % print_every == 0:
                train_loss /= print_every
                print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_idx * len(dataset), 
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), train_loss),
                      end='')
                train_losses.append(train_loss)
                train_loss = 0
        
        if val_loader is not None:
            targets, preds = get_predictions(model, val_loader)
            val_loss = get_loss(preds, targets)
            val_losses.append(val_loss)
            print('\tVal Loss: {:.6f}'.format(val_loss), end='')
            
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

######## OLD ##########

import torch.nn as nn
from torch import optim
from project.ranker.neural_ranker import RankNet
from scipy.stats import rankdata

class NeuralNetwork():
    def __init__(self, mf_sz, fs_sz, params):
        self.mf_sz, self.fs_sz = mf_sz, fs_sz
        self.latent_sz = params['latent_sz']
        self.epochs = params['epochs']
        self.lr = params['learning_rate']
        
        self.model = RankNet(mf_sz, fs_sz, self.latent_sz)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train(self, dl):
        train_losses = []
        for epoch in range(self.epochs):
            train_loss = 0
            for X, y, y_scores in dl:
                # for each dataset
                for X_, y_, y_scores_ in zip(X, y, y_scores):
                    X_cont, X_cats, y_long = wide2long(X_[None,:],
                                                       y_[None,:])
                    X_cats = torch.LongTensor(X_cats)
                    # precalculate preds
                    pred = self.model(X_cont, X_cats)
                    
                    # pick random fs method
                    i = np.random.randint(0, len(X_cont))
                    n, j = 0, 0
                    while True:
                        j = np.random.randint(0, len(X_cont))
                        if i == j:
                            continue
                            
                        # if there is discrepancy in ranking    
                        if pred[i] >= pred[j] and y_long[i] < y_long[j]:
                            break                            
                        elif pred[i] <= pred[j] and y_long[i] > y_long[j]:
                            i, j = j, i
                            break
                        
                        n += 1
                        if n == len(X_cont):
                            break
                    
                    # if r[i] >= r[j] wrongly
                    if n < len(X_cont):
                        train_loss += self.train_step(X_cont[[i,j]], X_cats[[i,j]], pred[[i,j]], n)

            train_losses.append(train_loss) 
        return train_losses
    
    def train_step(self, X_cont, X_cats, preds, n):
        self.model.train()
        self.optimizer.zero_grad()
        preds = self.model(X_cont, X_cats)
        
        loss = preds[1] - preds[0] + 1
        
#         loss = F.relu(preds[1] - preds[0] + 1)
#         loss = 2 * (1 / (1 + torch.exp(-loss))) + 1
#         loss = torch.log(loss + 1) / (n + 1)
        
        print(n, preds, loss)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict(self, dl):
        preds = []
        self.model.eval()
        for X, y, y_scores in dl:
            X_cont, X_cats, y_long = wide2long(X, y)
            X_cats = torch.LongTensor(X_cats)
            X_cont.requires_grad_(False)
            X_cats.requires_grad_(False)
            
            pred = self.model(X_cont, X_cats).cpu().detach().numpy()
            
            pred = np.array([rankdata(x, method='ordinal') for x in \
                             np.reshape(pred, y.shape)]).astype(int)
            preds.extend(pred)
        return np.array(preds)