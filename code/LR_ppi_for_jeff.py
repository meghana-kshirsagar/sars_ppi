
import sys
import pandas as pd
import numpy as np
#import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import train_test_split

from utils import do_logreg_paramtuning, normalize_train_test,impute_train_test,imputeX,get_aucpr,get_auc,binarize,get_fmax,get_aucpr_R,get_auc_R,compute_eval_measures,compute_early_prec,get_early_prec,compute_fmax


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
    X_pos = pd.read_csv(pos_feats_file, header=0)
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
    del X_neg_all

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
    y_train = y_train.ravel()
    clf = do_logreg_paramtuning(X_train, y_train, 0)
    print("Finished training ...")
    curr_perf = []
    y_pred = clf.predict(X_test)
    curr_perf += [metrics.accuracy_score(y_test, y_pred)]
    print(metrics.confusion_matrix(y_test, y_pred))
    y_pred = clf.predict_proba(X_test)
    curr_perf += [get_aucpr_R(y_test, y_pred[:,1])]
    curr_perf += [get_auc_R(y_test, y_pred[:,1])]
    print("Performance: ",curr_perf)

    # predict on larger set, output predictions
    print("Predicting on all test pairs now... ")
    X_neg_all = pd.read_csv(neg_feats_file, header=0)
    scores = (clf.predict_proba(X_neg_all))[:,1]
    neg_pps['score'] = scores   
    neg_pps.to_csv(outfile)
    
    # save model
    #save_model(clf,format("models/ebm_covonly_split%d_1to1_int.pkl" % split))
    
