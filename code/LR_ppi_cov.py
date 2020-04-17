
import pandas as pd
import numpy as np
#import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from interpret import show
from interpret.data import ClassHistogram
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression

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
    clf = clf.fit(X_train, y_train)
    return clf

def normalize_train_test(X_train, X_test, X_cov):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_cov = scaler.transform(X_cov)
    return X_train, X_test, X_cov

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


#home_dir = "/home/meghanak/projects/COVID/"

pos_index = range(294)
X_cov = pd.read_csv("features/test_cov2_pairs_feats.csv", header=0, index_col=0)
feat_names=X_cov.columns
#samp = np.random.randint(300,X_cov.shape[0],300)
#samp = np.concatenate((pos_index,samp))
#X_cov = X_cov.iloc[samp, :]
y_cov = np.zeros((X_cov.shape[0],1))
y_cov[pos_index]=1

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
    clf = do_logreg_paramtuning(X_train_cov, y_train_cov, 0)
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

print(np.mean(splitwise_perf,axis=0))


