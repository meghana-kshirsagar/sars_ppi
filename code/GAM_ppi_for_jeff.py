
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
    clf = clf.fit(X_train, y_train)
    return clf


if __name__ == "__main__":
    posfile = sys.argv[1]
    negfile = sys.argv[2]
    negfrac = float(sys.argv[3])
    ppis_file = sys.argv[4]
    pos_hprots_file = sys.argv[5]
 
    # read human proteins to select as positives
    krogan_ppis = pd.read_csv(ppis_file, header=0, index_col=0)
    print(krogan_ppis.head())
    with open(pos_hprots_file, 'r') as hpin:
        hprots_jeff = [line.strip() for line in hpin]
    print(hprots_jeff)
    print(krogan_ppis.shape)
    print(len(hprots_jeff))
    pick_idx = np.concatenate([np.where(krogan_ppis.iloc[:,1]==hprots_jeff[i])[0] for i in range(len(hprots_jeff))])
    print(pick_idx)

    # reading data
    print('Reading pos file... ')
    X_pos = pd.read_csv(posfile, compression='gzip', header=0)
    npos = X_pos.shape[0]
    X_train_pos = X_pos.iloc[pick_idx, :]
    X_test_pos = X_pos.drop(pick_idx)
    print('Reading neg file... ')
    #X_neg = pd.read_csv(negfile, compression='gzip', header=0)
    X_neg = pd.read_csv(negfile, header=0)
    nneg = X_neg.shape[0]
    feat_names=X_pos.columns
    # sample random negatives
    samp = np.random.randint(0,nneg,int(npos*negfrac))
    X_neg = X_neg.iloc[samp, :]
    nneg = X_neg.shape[0]

    # generate train/test splits
    X_train_neg, X_test_neg = train_test_split(X_neg, test_size=0.2)
    X_train = pd.DataFrame(np.row_stack((X_train_pos, X_train_neg)), columns=feat_names)
    X_test = pd.DataFrame(np.row_stack((X_test_pos, X_test_neg)), columns=feat_names)
    y_test = np.zeros((X_test.shape[0],1))
    y_train = np.zeros((X_train.shape[0],1))
    y_train[range(X_train_pos.shape[0])]=1
    y_test[range(X_test_pos.shape[0])]=1
    print("X size: ",X_train.shape[0],'x',X_train.shape[1])
    print("y size: ",y_train.shape[0],'x',y_train.shape[1])
    print("X-test size: ",X_test.shape[0],'x',X_test.shape[1])
    print("y-test size: ",y_test.shape[0],'x',y_test.shape[1])
    
    clf = tune_ebm(X_train, y_train)
    curr_perf = []
    y_pred = clf.predict(X_test)
    curr_perf += [metrics.accuracy_score(y_test, y_pred)]
    print(metrics.confusion_matrix(y_test, y_pred))
    y_pred = clf.predict_proba(X_test)
    curr_perf += [get_aucpr(y_test, y_pred[:,1])]
    curr_perf += [get_auc(y_test, y_pred[:,1])]
    print(curr_perf)
    # save model
    #save_model(clf,format("models/ebm_covonly_split%d_1to1_int.pkl" % split))
    
