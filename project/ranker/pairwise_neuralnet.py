import numpy as np
from scipy.stats import rankdata
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

class RankDataset(Dataset):
    def __init__(self, X, y, y_scores):
        super().__init__()
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.y_scores = y_scores.astype(np.float32)
        
        self.mf_sz = X.shape[1]
        self.fs_sz = y.shape[1]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx], self.y_scores[idx]]
    
class RankNet(nn.Module):
    def __init__(self, metafeatures_sz, featsel_methods_sz, latent_sz):
        super().__init__()
        self.metafeatures_sz = metafeatures_sz
        self.featsel_methods_sz = featsel_methods_sz
        
        self.linear = nn.Linear(metafeatures_sz, latent_sz)
        self.embedding = nn.Embedding(featsel_methods_sz, latent_sz)
        
        self.bn1 = nn.BatchNorm1d(latent_sz)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.1)
        
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

def wide2long(X, y):
    n_samples, n_classes = y.shape
    X_cont = np.repeat(X, n_classes, axis=0)
    X_cats = np.array(list(range(n_classes)) * n_samples)
    return X_cont, X_cats.astype(int), y.reshape(-1)        

class NeuralNetwork():
    def __init__(self, mf_sz, fs_sz, params):
        self.mf_sz, self.fs_sz = mf_sz, fs_sz
        self.latent_sz = params['latent_sz']
        self.epochs = params['epochs']
        self.lr = params['learning_rate']
        self.num_negative_samples = params['num_negative_samples']
        
        self.model = RankNet(mf_sz, fs_sz, self.latent_sz)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train(self, dl):
        train_losses = []
        for epoch in range(self.epochs):
            train_loss = 0
            for X, y, y_scores in dl:
                # for each dataset
                X_cont, X_cats, y_long = wide2long(X, y)
                X_cats = torch.LongTensor(X_cats)

                positive_pred = self.model(X_cont, X_cats)
                negative_pred = self.get_multiple_negative_preds(X_cont, n=self.num_negative_samples)
                
                train_loss += self.train_step(positive_pred, negative_pred)

            train_losses.append(train_loss) 
        return train_losses
    
    def get_negative_preds(self, X_cont):
        negative_items = np.random.randint(0, self.fs_sz, len(X_cont), dtype=np.int64)
        X_cats = torch.from_numpy(negative_items)
        return self.model(X_cont, X_cats)
    
    def get_multiple_negative_preds(self, X_cont, n=10):
        negative_preds = self.get_negative_preds(X_cont[None, ...] 
                                                 .expand(n, *X_cont.shape)
                                                 .reshape(-1, X_cont.shape[-1]))
        return negative_preds.view(n, len(X_cont))
    
    def train_step(self, positive_preds, negative_preds):
        self.model.train()
        self.optimizer.zero_grad()
        
        highest_negative_preds, _ = torch.max(negative_preds, 0)
        loss = torch.clamp(highest_negative_preds - positive_preds + 1.0, 0.0).mean()
        
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
    
from project.utils.metrics import evaluate_metric

def cv_neuralnet(X, y, y_scores, kfolds, params, verbose_folds=False):
    results = []
    models = []
    X = StandardScaler().fit_transform(X)
    for idx, (trn_idx, val_idx) in enumerate(kfolds.split(X, y)):
        X_trn, y_trn, y_scores_trn = X[trn_idx], y[trn_idx], y_scores[trn_idx]
        X_val, y_val, y_scores_val = X[val_idx], y[val_idx], y_scores[val_idx]
        
        trn_ds = RankDataset(X_trn, y_trn, y_scores_trn)
        val_ds = RankDataset(X_val, y_val, y_scores_val)
        
        neuralnet = NeuralNetwork(trn_ds.mf_sz, trn_ds.fs_sz, params)
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
        
#         break # 1-fold
          
    results = np.array(results)
    print()
    print(f'Trn_Spearman: {results[:,0].mean(): .4f} +/-{results[:,0].std():.4f} | '
          f'Val_Spearman: {results[:,1].mean(): .4f} +/-{results[:,1].std():.4f}\n'
          f'Trn_ACCLoss:  {results[:,2].mean(): .4f} +/-{results[:,2].std():.4f} | '
          f'Val_ACCLoss:  {results[:,3].mean(): .4f} +/-{results[:,3].std():.4f}')
    print()
    return results, models