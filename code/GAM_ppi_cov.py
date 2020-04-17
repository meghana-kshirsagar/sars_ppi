
import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split
#import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from interpret import show
from interpret.data import ClassHistogram
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV, StratifiedKFold

import pickle
from interpret.glassbox import ExplainableBoostingClassifier, LogisticRegression, ClassificationTree, DecisionListClassifier
from interpret.perf import ROC

seed=0

def normalize_train_test(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def impute_train_test(X_train, X_test):
    #replace -8888 values with Nan and then use simple imputer 
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(X_train)
    X_train = imp_mean.transform(X_train)
    X_test = imp_mean.transform(X_test)
    return X_train, X_test

def imputeX(X):
    #replace -8888 values with Nan and then use simple imputer 
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(X)
    X = imp_mean.transform(X)
    return X

def get_aucpr(y_true, y_pred, pos_label=1):
    #fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, pred, pos_label=2)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred, pos_label)
    auc_val = metrics.auc(recall, precision)
    return auc_val

def get_auc(labels, preds, pos_label=1):
    fpr, tpr, _ = metrics.roc_curve(labels, preds, pos_label)
    return metrics.auc(fpr, tpr)

def binarize(y_pred):
    return [int(x >= 0.5) for x in y_pred]


def save_model(ebm, model_file):
    model_pkl = open(model_file, 'wb')
    pickle.dump(ebm,model_pkl)
    model_pkl.close()


if __name__ == "__main__":
    posfile = sys.argv[1]
    negfile = sys.argv[2]
    negfrac = float(sys.argv[3])
    print('Reading pos file... ')
    X_pos = pd.read_csv(posfile, compression='gzip', header=0)
    npos = X_pos.shape[0]
    print('Reading neg file... ')
    X_neg = pd.read_csv(negfile, compression='gzip', header=0)
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
    
    # create cov splits
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    train_idxes_cov = []
    test_idxes_cov = []
    for train_index, test_index in kf.split(X_cov,y_cov):
        train_idxes_cov.append(train_index)
        test_idxes_cov.append(test_index)
    
    
    splitwise_perf = []
    for split in range(0,5):
        X_train_cov, X_test_cov = X_cov.iloc[train_idxes_cov[split],:], X_cov.iloc[test_idxes_cov[split],:]
        y_train_cov, y_test_cov = y_cov[train_idxes_cov[split]], y_cov[test_idxes_cov[split]]
        y_train_cov = y_train_cov.ravel()
        clf = ExplainableBoostingClassifier(random_state=seed) #, interactions=100)
        clf.fit(X_train_cov, y_train_cov)
        curr_perf = []
        y_pred_cov = clf.predict(X_test_cov)
        curr_perf += [metrics.accuracy_score(y_test_cov, y_pred_cov)]
        print(metrics.confusion_matrix(y_test_cov, y_pred_cov))
        y_pred_cov = clf.predict_proba(X_test_cov)
        curr_perf += [get_aucpr(y_test_cov, y_pred_cov[:,1])]
        curr_perf += [get_auc(y_test_cov, y_pred_cov[:,1])]
        print(curr_perf)
        splitwise_perf.append(curr_perf)
        # save model
        #save_model(clf,format("models/ebm_covonly_split%d_1to1_noint.pkl" % split))
    
    
    print(np.mean(splitwise_perf,axis=0))
    
