
import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split
#import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#from interpret import show
#from interpret.data import ClassHistogram
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold

import pickle
from interpret.glassbox import ExplainableBoostingClassifier
#from interpret.perf import ROC

seed=0

def do_logreg_paramtuning(X_train, y_train, class_wt):
    reslist = []
    metric_idx=1  # index where AUC is stored
    for cval in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 10**5]:
        logreg = LogisticRegression(random_state=0, penalty='l2', C=cval, max_iter=10000, solver='lbfgs')   # class_weight={0:(1-class_wt+0.1), 1:1}
        cv_results = cross_validate(logreg, X_train, y_train, cv=5, scoring='average_precision')
        reslist.append((cval, np.mean(cv_results['test_score'])))
    print(*reslist, sep='\n')
    reslist = np.asarray(reslist)
    bestid = np.where(reslist[:,metric_idx]==max(reslist[:,metric_idx]))[0][0]
    clf = LogisticRegression(random_state=0, penalty='l2', C=reslist[bestid,metric_idx], max_iter=10000, solver='lbfgs')    
    clf.fit(X_train, y_train)
    return clf

def normalize_train_test(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

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
   
    if len(sys.argv) < 8:
        print("Usage: <pos_feats_file>  <neg_feats_file>  <negatives-frac>  <ppis_file>  <all_pps_file>  <pos_hprots_file>  <outfile>\n")
        exit(1)

    pos_feats_file = sys.argv[1]
    neg_feats_file = sys.argv[2]
    negfrac = float(sys.argv[3])
    ppis_file = sys.argv[4]
    neg_pps_file = sys.argv[5]
    pos_hprots_file = sys.argv[6]
    outfile = sys.argv[7]
 
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

    # read negative protein pairs
    neg_pps = pd.read_csv(neg_pps_file, header=0, index_col=0)

    # reading features
    print('Reading pos file... ')
    X_pos = pd.read_csv(pos_feats_file, compression='gzip', header=0)
    npos = X_pos.shape[0]
    X_train_pos = X_pos.iloc[pick_idx, :]
    X_test_pos = X_pos.drop(pick_idx)
    print('Reading neg file... ')
    #X_neg = pd.read_csv(neg_feats_file, compression='gzip', header=0)
    X_neg_all = pd.read_csv(neg_feats_file, header=0)
    nneg = X_neg_all.shape[0]
    feat_names=X_pos.columns
    # sample random negatives
    samp = np.random.randint(0,nneg,int(npos*negfrac))
    X_neg = X_neg_all.iloc[samp, :]
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

    # train and test, performance output    
    #clf = tune_ebm(X_train, y_train)
    clf = do_logreg_paramtuning(X_train, y_train, 0)
    print("Finished training ...")
    curr_perf = []
    y_pred = clf.predict(X_test)
    curr_perf += [metrics.accuracy_score(y_test, y_pred)]
    print(metrics.confusion_matrix(y_test, y_pred))
    y_pred = clf.predict_proba(X_test)
    curr_perf += [get_aucpr(y_test, y_pred[:,1])]
    curr_perf += [get_auc(y_test, y_pred[:,1])]
    print("Performance: ",curr_perf)

    # predict on larger set, output predictions
    print("Predicting on all test pairs now... ")
    scores = (clf.predict_proba(X_neg_all))[:,1]
    neg_pps['score'] = scores   
    neg_pps.to_csv(outfile)
    
    # save model
    #save_model(clf,format("models/ebm_covonly_split%d_1to1_int.pkl" % split))
    
