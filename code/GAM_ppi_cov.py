
import sys
import pandas as pd
import numpy as np
#import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from interpret.glassbox import ExplainableBoostingClassifier  #, LogisticRegression, ClassificationTree

from utils import do_logreg_paramtuning, normalize_train_test,impute_train_test,imputeX,get_aucpr,get_auc,binarize,get_fmax,get_aucpr_R,get_auc_R,compute_eval_measures,compute_early_prec,get_early_prec,compute_fmax


def tune_ebm(X_train, y_train):
    reslist = []
    metric_idx=1  # index where AUC is stored
    for interac in [50, 100, 500]: 
        clf = ExplainableBoostingClassifier(random_state=seed, interactions=interac)
        cv_results = cross_validate(clf, X_train, y_train, cv=3, scoring='average_precision')
        reslist.append((interac, np.mean(cv_results['test_score'])))
    print(*reslist, sep='\n')
    reslist = np.asarray(reslist)
    bestid = np.where(reslist[:,metric_idx]==max(reslist[:,metric_idx]))[0][0]
    clf = ExplainableBoostingClassifier(random_state=seed, interactions=reslist[bestid,0])
    clf.fit(X_train, y_train)
    return clf


if __name__ == "__main__":
    posfile = sys.argv[1]
    negfile = sys.argv[2]
    negfrac = float(sys.argv[3])
    interac = int(sys.argv[4])
    print('Reading pos file... ')
    #X_pos = pd.read_csv(posfile, compression='gzip', header=0)
    X_pos = pd.read_csv(posfile, header=0)
    npos = X_pos.shape[0]
    print('Reading neg file... ')
    #X_neg = pd.read_csv(negfile, compression='gzip', header=0)
    X_neg = pd.read_csv(negfile, header=0)
    nneg = X_neg.shape[0]
    feat_names=X_pos.columns
    samp = np.random.randint(0,nneg,int(npos*negfrac))
    X_neg = X_neg.iloc[samp, :]
    nneg = X_neg.shape[0]
    X_cov = pd.DataFrame(np.row_stack((X_pos, X_neg)), columns=feat_names)
    y_cov = np.zeros((npos+nneg,1))
    y_cov[range(npos)]=1
    print("X size: ",X_cov.shape[0],'x',X_cov.shape[1])
    print("y size: ",y_cov.shape[0],'x',y_cov.shape[1])
    del X_neg
    
    # create cov splits
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    train_idxes_cov = []
    test_idxes_cov = []
    for train_index, test_index in kf.split(X_cov,y_cov):
        train_idxes_cov.append(train_index)
        test_idxes_cov.append(test_index)
    
    #for interac in [0]: # [5, 10, 50, 100, 300, 500]: 
    if True:
        print("======================== ", interac," ======================")
        splitwise_perf = []
        for split in range(0,5):
            X_train_cov, X_test_cov = X_cov.iloc[train_idxes_cov[split],:], X_cov.iloc[test_idxes_cov[split],:]
            y_train_cov, y_test_cov = y_cov[train_idxes_cov[split]], y_cov[test_idxes_cov[split]]
            y_train_cov = y_train_cov.ravel()
            #clf = tune_ebm(X_train_cov, y_train_cov)

            if interac==0:
                clf = ExplainableBoostingClassifier()
            else:
                clf = ExplainableBoostingClassifier(interactions=interac)

            clf.fit(X_train_cov, y_train_cov)
            curr_perf = []
            y_pred_cov = clf.predict(X_test_cov)
            #curr_perf += [metrics.accuracy_score(y_test_cov, y_pred_cov)]
            print(metrics.confusion_matrix(y_test_cov, y_pred_cov))
            y_pred_cov = clf.predict_proba(X_test_cov)
            curr_perf += [get_aucpr_R(y_test_cov, y_pred_cov[:,1])]
            curr_perf += [get_auc_R(y_test_cov, y_pred_cov[:,1])]
            curr_perf += [get_fmax(y_test_cov, y_pred_cov[:,1])]
            curr_perf += get_early_prec(y_test_cov, y_pred_cov[:,1])
            print(curr_perf)
            splitwise_perf.append(curr_perf)
            # save model
            #save_model(clf,format("models//ebm_covonly_split%d_1to10_int%d.pkl" % (split, interac)))
            #save_model(clf,format("models/100trials/ebm_covonly_split%d_1to1_int100_trial%d.pkl" % (split, interac)))
        print('             AUC-PR   ROC   F-MAX   EARLY-PREC@0.1  EARLY-PREC@0.2  EARLY-PREC@0.5') 
        print('[AVERAGE] ',np.mean(splitwise_perf,axis=0))
        
