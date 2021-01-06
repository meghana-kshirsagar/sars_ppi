
import sys
import time
import pandas as pd
import numpy as np
import pickle
#import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from interpret.glassbox import ExplainableBoostingClassifier  #, LogisticRegression, ClassificationTree

from utils import do_logreg_paramtuning, normalize_train_test,impute_train_test,imputeX,get_aucpr,get_auc,binarize,get_fmax,get_aucpr_R,get_auc_R,compute_eval_measures,compute_early_prec,get_early_prec,compute_fmax, save_model


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

# keep skew times as many negatives
def undersample_negatives(X, y, skew):
    npos = sum(y==1)
    negindices = np.where(y==0)[0]
    nneg = skew*npos
    negfrac = nneg/len(negindices)
    if(negfrac >= 1.0):
        return X, y
    print('len-neg:',len(negindices),' negfrac:',negfrac)
    negindices = negindices[np.where(np.random.sample(len(negindices)) >= negfrac)[0]] #remove fraction of elements
    keepindices = [i for i in range(0,X.shape[0]) if i not in negindices]
    X = X.iloc[keepindices, :]
    y = y[keepindices]
    return X, y

def subset_all_features(feats):
    feats_subset = pd.read_csv("~/projects/COVID/models/new_data/mixed_negatives/top_3mer_feats.txt",header=None)
    feats_subset = feats_subset.values
    feat_names = feats.columns
    feat_names_subset = [f for f in feat_names if f in feats_subset]
    feat_names = np.concatenate((feat_names[0:763], feat_names_subset))
    print('After subsetting: ',len(feat_names))
    feats = feats.loc[:,feat_names]
    print('After subsetting: ',feats.shape)
    return feats

def read_all_features(feats_file="features/all_proteinids_ctriad_123merfeats.pkl"):
    global allfeats
    st = time.time()
    allfeats = pd.read_pickle(feats_file)
    #allfeats = pd.read_csv(feats_file, header=0, index_col=0)
    print('Finished reading all features: ',allfeats.shape,' in time: ',(time.time()-st))
    #allfeats.to_pickle('features/all_proteinids_ctriad_123merfeats.pkl')
    allfeats = subset_all_features(allfeats)

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

    print('Reading neg file... ')
    ppis_neg = pd.read_csv(negfile, header=0, index_col=0)
    nneg = ppis_neg.shape[0]
    #samp = np.random.randint(0,nneg,int(npos*negfrac))

    samp1 = np.random.randint(0,6180,3000)
    samp = np.random.randint(6180,nneg,int(npos*negfrac)-2800)
    ppis_neg1 = ppis_neg.iloc[samp1, :]
    ppis_neg = ppis_neg.iloc[samp, :]
    ppis_neg = ppis_neg1.append(ppis_neg)

    #ppis_neg = ppis_neg.iloc[samp, :]
    X_neg = get_ppi_features(ppis_neg)
    nneg = X_neg.shape[0]

    feat_names = X_pos.columns
    X_cov = pd.DataFrame(np.row_stack((X_pos, X_neg)), columns=feat_names)
    y_cov = np.zeros((npos+nneg,1))
    y_cov[range(npos)]=1
    print("X size: ",X_cov.shape[0],'x',X_cov.shape[1])
    print("y size: ",y_cov.shape[0],'x',y_cov.shape[1])
    del allfeats, X_neg
    
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

            #X_train_cov, y_train_cov = undersample_negatives(X_train_cov, y_train_cov, 50)

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
            save_model(clf,format("%s/split%d_1to%d_int%d_trial%d.pkl" % (out_dir, split, int(negfrac), interac, trial)))
        print('             AUC-PR   ROC   F-MAX   EARLY-PREC@0.1  EARLY-PREC@0.2  EARLY-PREC@0.5') 
        print('[AVERAGE] ',np.mean(splitwise_perf,axis=0))
        
