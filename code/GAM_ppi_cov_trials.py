
import sys
import pandas as pd
import numpy as np
import time
#import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import pickle

from interpret.glassbox import ExplainableBoostingClassifier  #, LogisticRegression, ClassificationTree

from utils import do_logreg_paramtuning, normalize_train_test,impute_train_test,imputeX,get_aucpr,get_auc,binarize,get_fmax,get_aucpr_R,get_auc_R,compute_eval_measures,compute_early_prec,get_early_prec,compute_fmax,save_model


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


def read_all_features(feats_file="features/all_proteinids_ctriad_123merfeats.pkl"):
    global allfeats
    st = time.time()
    allfeats = pd.read_pickle(feats_file)
    #allfeats = pd.read_csv(feats_file, header=0, index_col=0)
    print('Finished reading all features: ',allfeats.shape,' in time: ',(time.time()-st))
    #allfeats.to_pickle('features/all_proteinids_ctriad_123merfeats.pkl')
    #allfeats = subset_all_features(allfeats)

def get_ppi_features(ppi_list):
    print(ppi_list.head())
    prot1 = list(ppi_list.iloc[:,0])
    prot2 = list(ppi_list.iloc[:,1])
    rows_present = [ii for ii in range(len(prot1)) if ((prot1[ii] in allfeats.index) and (prot2[ii] in allfeats.index))]
    print('Total ppis present: ',len(rows_present))
    feats1 = allfeats.loc[[prot1[i] for i in rows_present]]
    feats2 = allfeats.loc[[prot2[i] for i in rows_present]]
    print(feats1.shape)
    print(feats2.shape)
    feat_names = list(feats1) + [x+".h" for x in feats2.columns]
    print(feat_names[-1])
    print('Feats len: ',len(feat_names))
    ppifeats = pd.DataFrame(np.column_stack((feats1, feats2)), columns = feat_names)
    print(ppifeats.shape)

    new_ppi_list = ppi_list.iloc[rows_present]
    new_ppi_list.to_csv('data/sars_cov2_human_ppi_good.csv')

    return ppifeats

allfeats = []


if __name__ == "__main__":
    posfile = sys.argv[1]
    negfile = sys.argv[2]
    negfrac = float(sys.argv[3])
    interac = int(sys.argv[4])
    trial = int(sys.argv[5])
    out_dir = sys.argv[6]

    read_all_features()
    print('Reading pos file... ')
    ppis_pos = pd.read_csv(posfile, header=0, index_col=0)
    X_pos = get_ppi_features(ppis_pos)
    npos = X_pos.shape[0]
    sys.exit(0)

    print('Reading neg file... ')
    ppis_neg = pd.read_csv(negfile, header=0, index_col=0)
    nneg = ppis_neg.shape[0]
    samp = np.random.randint(0,nneg,int(npos*negfrac))
    ppis_neg = ppis_neg.iloc[samp, :]
    X_neg = get_ppi_features(ppis_neg)
    nneg = X_neg.shape[0]

    feat_names = X_pos.columns
    X_cov = pd.DataFrame(np.row_stack((X_pos, X_neg)), columns=feat_names)
    y_cov = np.zeros((npos+nneg,1))
    y_cov[range(npos)]=1
    print("X size: ",X_cov.shape[0],'x',X_cov.shape[1])
    print("y size: ",y_cov.shape[0],'x',y_cov.shape[1])
    #del X_neg
    
    #for interac in [0]: # [5, 10, 50, 100, 300, 500]: 
    if True:
        print("======================== ", interac," ======================")
        if interac==0:
            clf = ExplainableBoostingClassifier()
        else:
            clf = ExplainableBoostingClassifier(interactions=interac)

        clf.fit(X_cov, y_cov)
        # test on everything
        #X_neg = pd.read_csv(negfile, header=0)
        X_cov = pd.DataFrame(np.row_stack((X_pos, X_neg_all)), columns=feat_names)
        print('Predicting on #examples:', X_cov.shape[0])
        y_pred = clf.predict_proba(X_cov)
        y_pred = y_pred[:,1]
        np.save(format("%s/int%d_trial%d_preds.npy") % (out_dir, interac,trial), y_pred)
        save_model(clf,format("%s/int%d_trial%d.pkl" % (out_dir, interac, trial)))
        
