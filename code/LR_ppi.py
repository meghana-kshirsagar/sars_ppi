
import pandas as pd
import numpy as np
#import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from interpret import show
from interpret.data import ClassHistogram
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression

seed=0


def do_logreg_paramtuning(X_train, y_train, class_wt):
    reslist = []
    metric_idx=1  # index where AUC is stored
    for cval in [0.1, 1, 10]: # 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 10**5]:
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


home_dir = "/home/meghanak/projects/COVID/"
X_pos = pd.read_csv(home_dir+"features/training_ppis_feats_humanpartners.csv", header=0, index_col=0)
samp = np.where(np.random.sample(X_pos.shape[0]) < 0.45)[0]
X_pos = X_pos.iloc[samp, :]
X_neg = pd.read_csv(home_dir+"features/training_negs_feats.csv", header=0, index_col=0)
feat_names=X_neg.columns
npos = X_pos.shape[0]
nneg = X_neg.shape[0]

print("#pos: ",npos," #neg: ",nneg)
y = np.vstack((np.ones((npos,1)), np.zeros((nneg,1))))
print(y.shape)
X = pd.DataFrame(np.row_stack((X_pos, X_neg)), columns=feat_names)

#del X_pos, X_neg

X_cov = pd.read_csv("features/test_cov2_pairs_feats.csv", header=0, index_col=0)
y_cov = np.zeros((X_cov.shape[0],1))
y_cov[0:294]=1


kf = KFold(n_splits=5, shuffle=True)
train_idxes = []
test_idxes = []
for train_index, test_index in kf.split(X):
    train_idxes.append(train_index)
    test_idxes.append(test_index)

splitwise_perf = []
for split in range(0,5):
    X_train, X_test = X.iloc[train_idxes[split],:], X.iloc[test_idxes[split],:]
    y_train, y_test = y[train_idxes[split]], y[test_idxes[split]]
    #X_train, X_test, X_cov = normalize_train_test(X_train, X_test, X_cov)
    clf = do_logreg_paramtuning(X_train, y_train, 0)
    y_pred = clf.predict(X_test)
    print(metrics.confusion_matrix(y_test, y_pred))
    curr_perf = []
    curr_perf += [metrics.accuracy_score(y_test, y_pred)]
    y_pred = clf.predict_proba(X_test)
    curr_perf += [get_aucpr(y_test, y_pred[:,1])]
    curr_perf += [get_auc(y_test, y_pred[:,1])]
    y_pred_cov = clf.predict_proba(X_cov)
    curr_perf += [get_aucpr(y_cov, y_pred_cov[:,1])]
    curr_perf += [get_auc(y_cov, y_pred_cov[:,1])]
    print(curr_perf)
    splitwise_perf.append(curr_perf)
   


print(splitwise_perf)


