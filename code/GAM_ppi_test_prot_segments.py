
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
import time
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
    clf.fit(X_train, y_train)
    return clf

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


def subset_all_features(feats):
    feats_subset = pd.read_csv("data/top_3mer_feats.txt",header=None)
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
    #allfeats = pd.read_pickle(feats_file)
    allfeats = pd.read_csv(feats_file, header=0, index_col=0)
    print('Finished reading all features: ',allfeats.shape,' in time: ',(time.time()-st))
    #allfeats.to_pickle('features/all_proteinids_ctriad_123merfeats.pkl')
    allfeats = subset_all_features(allfeats)


allfeats = []


if __name__ == "__main__":
    model_file = sys.argv[1]
    test_file = sys.argv[2]
    feat_file = sys.argv[3]
    out_file = sys.argv[4]
    print('Reading model file... ')
    clf = pickle.load(open(model_file, 'rb'))

    read_all_features(feat_file)
    print('Reading test file... ')
    ppis_test = pd.read_csv(test_file, header=None, index_col=0)
    X_test = get_ppi_features(ppis_test)
    print(X_test.shape)
    ntest = X_test.shape[0]
    y_test = np.zeros((ntest,1))
    
    y_pred = clf.predict(X_test)
    print(metrics.confusion_matrix(y_test, y_pred))
    y_pred = clf.predict_proba(X_test)
    y_pos = y_pred[:,1]
    print("\n".join([format("%s %s %g" % (x,y,z)) for (x,y,z) in zip(ppis_test.iloc[:,0], ppis_test.iloc[:,1], y_pos)]))
    with open(out_file, 'w') as filehandle:
        filehandle.writelines("%s\n" % val for val in y_pred[:,1])


