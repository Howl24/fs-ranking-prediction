import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

from ..utils.metrics import evaluate_metric


class RankDataset(Dataset):
    def __init__(self, X_cont, X_cats, y):
        super().__init__()
        self.X_cont = X_cont.astype(np.float32)
        self.X_cats = X_cats.astype(np.int64)
        self.y = y.astype(np.float32) # 0-1
        
        self.mf_sz = X_cont.shape[1]
        self.fs_sz = len(np.unique(X_cats))
        
    def __len__(self):
        return len(self.X_cont)
    
    def __getitem__(self, idx):
        return [self.X_cont[idx], self.X_cats[idx], self.y[idx]]

    
class RankNet(nn.Module):
    def __init__(self, metafeatures_sz, featsel_methods_sz, latent_sz, random_seed=42):
        super().__init__()
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        
        self.metafeatures_sz = metafeatures_sz
        self.featsel_methods_sz = featsel_methods_sz
        
        self.linear = nn.Linear(metafeatures_sz, latent_sz)
        self.embedding = nn.Embedding(featsel_methods_sz, latent_sz)
        
        self.bn1 = nn.BatchNorm1d(latent_sz)
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.3)
        
        self.lin3 = nn.Linear(latent_sz, 15)
        self.bn3 = nn.BatchNorm1d(15)
        self.drop3 = nn.Dropout(0.3)
        
        self.lin4 = nn.Linear(15, 1)
        
        self.emb_init(self.embedding)
        nn.init.kaiming_normal_(self.linear.weight.data)
        
    def forward(self, metafeatures, featsel_method):
        latent_metafeatures = self.linear(metafeatures)
#         latent_metafeatures = self.drop1(self.bn1(F.relu(latent_metafeatures)))
        
        latent_featsel_method = self.embedding(featsel_method)
#         latent_featsel_method = self.drop2(latent_featsel_method)      
        
        output = (latent_metafeatures * latent_featsel_method).sum(1)
#         output = self.drop3(self.bn3(F.relu(self.lin3(output))))
#         output = self.lin4(output)
#         return output
        return torch.sigmoid(output) # * (self.featsel_methods_sz - 1) + 1
    
    def emb_init(self, x):
        x = x.weight.data
        sc = 2 / (x.size(1) + 1)
        x.uniform_(-sc, sc)    
    
    
class NeuralNetwork():
    def __init__(self, mf_sz, fs_sz, params, USE_CUDA=False):
        self.mf_sz, self.fs_sz = mf_sz, fs_sz
        self.latent_sz = params['latent_sz']
        self.epochs = params['epochs']
        self.lr = params['learning_rate']
        self.USE_CUDA = USE_CUDA
        
        self.model = self.to_gpu(RankNet(mf_sz, fs_sz, self.latent_sz))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train(self, dl):
        train_losses = []
        for epoch in range(self.epochs):
            train_loss = 0
            for X_cont, X_cats, y in dl:
                X_cont, X_cats, y = self.to_gpu(X_cont, X_cats, y)
                train_loss += self.train_step(X_cont, X_cats, y)
            train_losses.append(train_loss) 
        return train_losses
    
    def train_step(self, X_cont, X_cats, y):
        self.model.train()
        self.optimizer.zero_grad()
        preds = self.model(X_cont, X_cats)
        loss = self.criterion(preds.view(-1), y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict(self, dl):
        preds = []
        self.model.eval()
        with torch.no_grad():
            for X_cont, X_cats, y in dl:
                X_cont, X_cats, y = self.to_gpu(X_cont, X_cats, y)
                pred = self.model(X_cont, X_cats).cpu().detach().numpy()
                preds.extend(pred)
        return np.array([rankdata(x, method='ordinal') for x in \
                         np.reshape(preds, (-1, self.fs_sz))]).astype(int)
    
    def to_gpu(self, *tensors):
        if self.USE_CUDA:
            tensors = [t.cuda() for t in tensors]
        if len(tensors) == 1:
            return tensors[0]
        return tensors
        
    
def wide2long(X, y):
    n_samples, n_classes = y.shape
    X_cont = np.repeat(X, n_classes, axis=0)
    X_cats = np.array(list(range(n_classes)) * n_samples)
    return X_cont, X_cats.astype(int), y.reshape(-1)
    
    
def cv_neuralnet(X, y, y_scores, kfolds, params, verbose_folds=False, 
                 USE_CUDA=False):
    results = []
    models = []
    X = StandardScaler().fit_transform(X)
    y = (y - y.min()) / (y.max() - y.min())
    for idx, (trn_idx, val_idx) in enumerate(kfolds.split(X, y)):
        X_trn, y_trn, y_scores_trn = X[trn_idx], y[trn_idx], y_scores[trn_idx]
        X_val, y_val, y_scores_val = X[val_idx], y[val_idx], y_scores[val_idx]
        
        X_trn_cont, X_trn_cats, y_trn_long = wide2long(X_trn, y_trn)
        X_val_cont, X_val_cats, y_val_long = wide2long(X_val, y_val)
        
        trn_ds = RankDataset(X_trn_cont, X_trn_cats, y_trn_long)
        val_ds = RankDataset(X_val_cont, X_val_cats, y_val_long)
        
        neuralnet = NeuralNetwork(trn_ds.mf_sz, trn_ds.fs_sz, params, USE_CUDA)
        trn_dl = DataLoader(trn_ds, batch_size=params['batch_sz'], shuffle=True)
        neuralnet.train(trn_dl)
        
        trn_dl = DataLoader(trn_ds, batch_size=params['batch_sz'], shuffle=False)
        val_dl = DataLoader(val_ds, batch_size=params['batch_sz'], shuffle=False)
        
        y_pred_trn = neuralnet.predict(trn_dl)
        y_pred_val = neuralnet.predict(val_dl)
        
        trn_spearman = evaluate_metric("spearman", y_trn, y_pred_trn)
        trn_acc_loss = evaluate_metric("mean_acc_loss", y_scores_trn, y_pred_trn)
        val_spearman = evaluate_metric("spearman", y_val, y_pred_val)
        val_acc_loss = evaluate_metric("mean_acc_loss", y_scores_val, y_pred_val)
        
        if verbose_folds:
            print(f'Fold {idx + 1:>3} | '
                  f'Trn_Spearman: {trn_spearman: .4f} | '
                  f'Val_Spearman: {val_spearman: .4f} | '
                  f'Trn_ACCLoss: {trn_acc_loss: .4f} | '
                  f'Val_ACCLoss: {val_acc_loss: .4f}')
            
        results.append((trn_spearman, val_spearman, 
                        trn_acc_loss, val_acc_loss))
        models.append(neuralnet)
          
    results = np.array(results)
    print()
    print(f'Trn_Spearman: {results[:,0].mean(): .4f} +/-{results[:,0].std():.4f} | '
          f'Val_Spearman: {results[:,1].mean(): .4f} +/-{results[:,1].std():.4f}\n'
          f'Trn_ACCLoss:  {results[:,2].mean(): .4f} +/-{results[:,2].std():.4f} | '
          f'Val_ACCLoss:  {results[:,3].mean(): .4f} +/-{results[:,3].std():.4f}')
    print()
    return results, models
    