import numpy as np
from scipy.stats import rankdata
from project.utils.metrics import evaluate_metric

def wide2long(X, y):
    n_samples, n_classes = y.shape
    X_cont = np.repeat(X, n_classes, axis=0)
    X_cats = np.array(list(range(n_classes)) * n_samples)[:, None]
    return np.concatenate([X_cont, X_cats], axis=1), y.reshape(-1)

def cv_lgbm(lightgbm, X, y, y_scores, kfolds, params, num_rounds=1000, 
            early_stopping_rounds=30, verbose_eval=False, metric='ndcg@13'):
    results = []
    models = []
    for idx, (trn_idx, val_idx) in enumerate(kfolds.split(X, y)):
        X_val, y_val = X[val_idx], y[val_idx]
        X_trn, y_trn = X[trn_idx], y[trn_idx]
        y_scores_trn = y_scores[trn_idx]
        y_scores_val = y_scores[val_idx]
        
        X_trn_long, y_trn_long = wide2long(X_trn, y_trn)
        X_val_long, y_val_long = wide2long(X_val, y_val)
        
        dtrn = lightgbm.Dataset(X_trn_long, y_trn_long, 
                                group=[y_trn.shape[1]] * y_trn.shape[0])
        dval = lightgbm.Dataset(X_val_long, y_val_long, 
                                group=[y_val.shape[1]] * y_val.shape[0])
        
        bst = lightgbm.train(params, dtrn, num_rounds, [dtrn, dval],
                             early_stopping_rounds=early_stopping_rounds,
                             verbose_eval=verbose_eval)
        
        y_pred_trn = np.array([rankdata(bst.predict(wide2long(x[None,:],
                                                              y[None,:])[0]), 
                               method='ordinal') for x, y in zip(X_trn, y_trn)])
        y_pred_val = np.array([rankdata(bst.predict(wide2long(x[None,:], 
                                                              y[None,:])[0]),
                               method='ordinal') for x, y in zip(X_val, y_val)])

        trn_spearman = evaluate_metric("spearman", y_trn, y_pred_trn)
        trn_acc_loss = evaluate_metric("mean_acc_loss", y_scores_trn, 
                                       y_trn.shape[1] - y_pred_trn + 1)
        val_spearman = evaluate_metric("spearman", y_val, y_pred_val)
        val_acc_loss = evaluate_metric("mean_acc_loss", y_scores_val, 
                                       y_val.shape[1] - y_pred_val + 1)
        trn_ndcg = bst.best_score['training'][metric]
        val_ndcg = bst.best_score['valid_1'][metric]
        
        print(f'Fold {idx + 1:>3} | '
              f'#Est: {bst.best_iteration:>3} | '
              f'Trn_Spearman: {trn_spearman: .4f} | '
              f'Val_Spearman: {val_spearman: .4f} | '
              f'Trn_ACCLoss: {trn_acc_loss: .4f} | '
              f'Val_ACCLoss: {val_acc_loss: .4f} | '
              f'Trn_NDCG: {trn_ndcg: .4f} | '
              f'Val_NDCG: {val_ndcg: .4f}')
        results.append((trn_spearman, val_spearman, 
                        trn_acc_loss, val_acc_loss,
                        trn_ndcg, val_ndcg))
        models.append(bst)
          
    results = np.array(results)
    print()
    print(f'Trn_Spearman: {results[:,0].mean(): .4f} +/-{results[:,0].std():.4f} | '
          f'Val_Spearman: {results[:,1].mean(): .4f} +/-{results[:,1].std():.4f}\n'
          f'Trn_ACCLoss:  {results[:,2].mean(): .4f} +/-{results[:,2].std():.4f} | '
          f'Val_ACCLoss:  {results[:,3].mean(): .4f} +/-{results[:,3].std():.4f}\n'
          f'Trn_NDCG:     {results[:,4].mean(): .4f} +/-{results[:,4].std():.4f} | '
          f'Val_NDCG:     {results[:,5].mean(): .4f} +/-{results[:,5].std():.4f}')
    print()
    return results, models