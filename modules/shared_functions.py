import itertools
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from collections import Counter
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, f1_score, log_loss, precision_recall_curve, average_precision_score, precision_recall_curve, recall_score, precision_score
#from sklearn.metrics import balanced_accuracy_score

import numpy as np
import pandas as pd
import time

plt.style.use('seaborn')

class DummySampler:

    def sample(self, X, y):
        return X, y

    def fit(self, X, y):
        return self

    def fit_resample(self, X, y):
        return self.sample(X, y)

def OHE(df, col = ['city','registered_via']):
   
    print('Categorical columns in dataset', col)
    
    c2,c3 = [],{}
    for c in col:
        if df[c].nunique()>2 :
            c2.append(c)
            c3[c] = 'ohe_'+c
    
    df = pd.get_dummies(df, columns=c2, drop_first=True, prefix=c3)
    return df

def balanced_accuracy(y_true, y_pred):
    # (TP/P + TN/N) / 2
    conf_m = confusion_matrix(y_true, y_pred)
    tn = conf_m[0][0]
    fn = conf_m[1][0]
    tp = conf_m[1][1]
    fp = conf_m[0][1]

    p_recall = tp / (tp + fn)   
    n_recall = tn / (tn + fp)
    print("P_Recall: " + str(round(p_recall, 3)) + "; N_Recall: " + str(round(n_recall, 3)))

    return (p_recall + n_recall) / 2

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

def normalise(train_set, test_set):
    print("Applying normalisation to train and test sets")
    scale_columns = ['bd', 'age_cat', 'registration_init_time', 'total_actual_payment', 'mean_payment_each_transaction', \
                     'plan_net_worth', 'total_order', 'cancel_times']  
 
    scale_us = intersection(train_set.columns, scale_columns) 
    
    scaler = preprocessing.StandardScaler().fit(train_set[scale_us])

    train_set[scale_us] = scaler.transform(train_set[scale_us])
    test_set[scale_us] = scaler.transform(test_set[scale_us])
    
    return train_set, test_set

def oversample_dataset(df, target_col = 'is_churn', sample_ratio = 1.0, r_state = None):
    train_negative = df[df[target_col] == 0]
    # Upsample minority class
    print("Sample ratio: ", sample_ratio, " :: ", int(train_negative.shape[0] * sample_ratio))
    train_positive_upsample = resample(df[df[target_col]==1], 
                                       replace = True, # sample with replacement
                                       n_samples = int(train_negative.shape[0] * sample_ratio), # to match majority class
                                       random_state = r_state) # reproducible results

    # Combine majority class with upsampled minority class
    return pd.concat([train_negative, train_positive_upsample])

def undersample_dataset(df, target_col = 'is_churn', r_state = None):
    # Down Sample Majority class
    down_sampled = resample(df[df[target_col] == 0], 
                           replace = True, # sample with replacement
                           n_samples = df[df[target_col]==1].shape[0], # to match minority class
                           random_state = r_state) # reproducible results

    # Combine majority class with upsampled minority class
    return pd.concat([df[df[target_col]==1], down_sampled])

def prepare_train_test_split(df, target_index, sampler = None, split_ratio = 0.7, r_state = None, cat_col = ['city','registered_via']): 
        
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,target_index+1:], df.iloc[:,target_index],
                                                        train_size=split_ratio, 
                                                        stratify=df.iloc[:,target_index],
                                                        random_state = r_state)
    
    if sampler: 
        print("PRE-SAMPLING:", X_train.shape, y_train.shape, Counter(y_train))
        ## This will fit the provided sampling approach to the training data 
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        print("POST-SAMPLING:", X_train.shape, y_train.shape, Counter(y_train))
    
    # Reconsctuct the dataframes
    X_train = pd.DataFrame(X_train, columns=df.iloc[:,0+1:].columns)
    X_test = pd.DataFrame(X_test, columns=df.iloc[:,0+1:].columns)
    y_train = pd.DataFrame(y_train, columns=['is_churn'])
    y_test = pd.DataFrame(y_test, columns=['is_churn'])
    
    # Normalise the dataset
    X_train, X_test = normalise(X_train, X_test)
    
    # One-hot-encode the categorical features
    X_train = OHE(X_train, cat_col)
    X_test = OHE(X_test, cat_col)
    
    print("X_train: ", X_train.shape, y_train.shape)
    print("X_test: ", X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test

def perform_experiment(df, classifier_set, sampler, iterations = 1, r_state = None, cv_iter = None, cat_col = ['city','registered_via']):
    metrics_all = pd.DataFrame()

    for i in range(0, iterations):

        print("Model Build Iteration", i)

        X_train, X_test, y_train, y_test = \
                    prepare_train_test_split(df, 
                                             df.columns.get_loc("is_churn"), 
                                             sampler = sampler[1], 
                                             split_ratio = 0.7, 
                                             r_state = r_state,
                                             cat_col = cat_col)
        
        if i == 0:
            plt.subplot(1, 2, 1)
            y_train.is_churn.value_counts().plot(kind='bar', title='Train Set - Count (target)')
            plt.subplot(1, 2, 2)
            y_test.is_churn.value_counts().plot(kind='bar', title='Test Set - Count (target)')
            plt.tight_layout()
            plt.show()

        model_build_results = train_model(X_train, X_test, 
                                          y_train.is_churn.values, y_test.is_churn.values, 
                                          classifiers = classifier_set,
                                          sampling_method = sampler[0],
                                          cv_iter = cv_iter
                                         )

        metrics = model_build_results[0]
        metrics['sample'] = i
        metrics_all = metrics_all.append(metrics)

        if i == 0:
            plot_data = []
            for res in model_build_results[1]:
                model_name, model, (fpr, tpr, roc_auc), (precision, recall, prc_auc) = res
                plot_data.append((model_name, tpr, fpr, roc_auc, precision, recall, prc_auc))   

            plot_roc_prc(plot_data)

    return metrics_all

def prepare_train_test_split_v1(df, target_index, sampling_type = None, sample_ratio = 1.0, split_ratio = 0.7, r_state = None): 
        
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,target_index+1:], df.iloc[:,target_index],
                                                        train_size=split_ratio, 
                                                        stratify=df.iloc[:,target_index],
                                                        random_state = r_state)
    
    train_set = X_train
    train_set['is_churn'] = y_train
    
    ## Do any over/under sampling if required
    if sampling_type == 'over':
        train_set = oversample_dataset(train_set, sample_ratio = sample_ratio, r_state=r_state).sort_values(['registration_init_time_dt'])
    elif sampling_type == 'under':
        train_set = undersample_dataset(train_set, r_state = r_state).sort_values(['registration_init_time_dt'])

    ## Generate the train and test datasets for this..  
    X_train, y_train = train_set.iloc[:, 0:-1], train_set.iloc[:, -1]
    
    # Normalise the dataset
    X_train, X_test = normalise(X_train, X_test)

    return X_train, X_test, y_train, y_test

def clean_dataset(df):
    return df.dropna().copy()

def train_model_v1(df, sampling_method = None, sample_ratio = 1.0, classifiers = [('RF', RandomForestClassifier(), {})]):
    
    target_index = 0
    results = []
    metric_cols = ["classifier", "sampling_method", "sampling_ratio", "accuracy", "precision", "recall", "f1_score", "log_loss", "time_taken", "aucroc", "auprc", "bal_acc"]
    
    metrics = pd.DataFrame(columns=metric_cols)

    X_train, X_test, y_train, y_test = prepare_train_test_split_v1(df, 0, sampling_method, sample_ratio)
    
    for name, model, params, metric in classifiers:
        print('Building {0} classifier'.format(name))
        start = time.time()
        
        if params:
            print("Optimising using GridSearchCV")
            clf = GridSearchCV(model, params, cv=5, verbose=2, scoring=metric, n_jobs=-1)
            clf.fit(X_train.drop(columns=['registration_init_time', 'registration_init_time_dt'], axis=1), 
                    y_train)         
            model = clf.best_estimator_ 
        else:
            print("No params set, using Standard training")
            model.fit(X_train.drop(columns=['registration_init_time', 'registration_init_time_dt'], axis=1), 
                    y_train)      
            
        y_predict = model.predict(X_test.drop(columns=['registration_init_time', 'registration_init_time_dt'], axis=1))
        y_predict_prob = model.predict_proba(X_test.drop(columns=['registration_init_time', 'registration_init_time_dt'], axis=1))[:,1]

        finish = time.time()
        
        # Compute ROC/AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_predict_prob)
        roc_auc = auc(fpr, tpr)        
        # Compute Precision-Recall
        precision, recall, thresholds = precision_recall_curve(y_test, y_predict_prob)
        # calculate precision-recall AUC
        auc_prc = auc(recall, precision)
        # average = weighted|macro|micro|samples
        pr_ap = average_precision_score(y_test, y_predict, average='weighted')

        metric_entry = pd.DataFrame([[model.__class__.__name__, sampling_method, sample_ratio if sampling_method else None,
                                      accuracy_score(y_test, y_predict),
                                      precision_score(y_test, y_predict),
                                      recall_score(y_test, y_predict),
                                      f1_score(y_test, y_predict), 
                                      log_loss(y_test, y_predict),
                                      finish-start,
                                      roc_auc, 
                                      auc_prc,
                                      balanced_accuracy(y_test, y_predict)]], 
                                    columns=metric_cols)
        metrics = metrics.append(metric_entry)
        
        results.append((model.__class__.__name__, model, (fpr, tpr, roc_auc), (precision, recall, auc_prc)))
        
    return (metrics, results)

def train_model(X_train, X_test, y_train, y_test, 
                classifiers = [('RF', RandomForestClassifier(), {})], 
                sampling_method = 'None', 
                cv_iter=None):
    
    target_index = 0
    results = []
    metric_cols = ["classifier", "sampling_method", "tn", "fn", "tp", "fp", "accuracy", "precision", "recall", "f1_score", "log_loss", "time_taken", "aucroc", "auprc", "bal_acc", "cv_score_mean", "cv_score_std"]
    metrics = pd.DataFrame(columns=metric_cols)
    
    for name, model, params, metric in classifiers:
        print('Building {0} classifier'.format(name))
        params = params.copy()
        start = time.time()
        
        if params:
            if params.pop('search_type', 'GRID_SEARCH_CV') == 'RANDOM_SEARCH_CV':
                print("Optimising using RandomizedSearchCV")
                clf = RandomizedSearchCV(model, params, cv=(cv_iter if cv_iter else 3), verbose=2, scoring=metric, n_jobs=-1)
            else:
                print("Optimising using GridSearchCV")
                clf = GridSearchCV(model, params, cv=(cv_iter if cv_iter else 3), verbose=2, scoring=metric, n_jobs=-1)
                
            clf.fit(X_train.drop(columns=['registration_init_time', 'registration_init_time_dt'], axis=1, errors='ignore'), 
                    y_train)         
            model = clf.best_estimator_ 
            print("CLF:", clf.best_params_)
        else:
            print("No params set, using Standard training")
            model.fit(X_train.drop(columns=['registration_init_time', 'registration_init_time_dt'], axis=1, errors='ignore'), 
                    y_train)      
            
        y_predict = model.predict(X_test.drop(columns=['registration_init_time', 'registration_init_time_dt'], axis=1, errors='ignore'))
        y_predict_prob = model.predict_proba(X_test.drop(columns=['registration_init_time', 'registration_init_time_dt'], axis=1, errors='ignore'))[:,1]
        
        finish = time.time()
        
        # Compute ROC/AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_predict_prob)
        roc_auc = auc(fpr, tpr)        
        # Compute Precision-Recall
        precision, recall, thresholds = precision_recall_curve(y_test, y_predict_prob)
        # calculate precision-recall AUC
        auc_prc = auc(recall, precision)
        # average = weighted|macro|micro|samples
        pr_ap = average_precision_score(y_test, y_predict, average='weighted')        

        if cv_iter:   
            print("Performing {}-fold CV on test set using {} metric".format(cv_iter, metric))
            cv_score = cross_val_score(model,
                                       X_test.drop(columns=['registration_init_time', 'registration_init_time_dt'], axis=1, errors='ignore'), 
                                       y_test, 
                                       scoring = metric, 
                                       n_jobs = -1,
                                       cv=cv_iter)
        
        conf_m = confusion_matrix(y_test, y_predict)
        tn = conf_m[0][0]
        fn = conf_m[1][0]
        tp = conf_m[1][1]
        fp = conf_m[0][1]

        metric_entry = pd.DataFrame([[model.__class__.__name__, 
                                      sampling_method,
                                      tn, fn, tp, fp,
                                      accuracy_score(y_test, y_predict),
                                      precision_score(y_test, y_predict),
                                      recall_score(y_test, y_predict),
                                      f1_score(y_test, y_predict), 
                                      log_loss(y_test, y_predict),
                                      finish-start,
                                      roc_auc, 
                                      auc_prc,
                                      balanced_accuracy(y_test, y_predict),
                                      np.mean(cv_score) if cv_iter else -1, 
                                      np.std(cv_score) if cv_iter else -1]], 
                                    columns=metric_cols)       
        
        metrics = metrics.append(metric_entry)
        
        results.append((model.__class__.__name__, model, (fpr, tpr, roc_auc), (precision, recall, auc_prc)))
        
    ## Compute the monetary cost of the trained churn models
    metrics['model_churn_cost'] = metrics.apply(
                                calc_churn_monetary_value, 
                                axis = 1, 
                                args = [len(y_test), np.mean(y_test)])    
        
    return (metrics, results)

def plot_roc(test_y, probs, title='ROC Curve', threshold_selected=None):
    """Plot an ROC curve for predictions. 
       Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py"""

    tpr, fpr, threshold = roc_curve(test_y, probs)
    plt.figure(figsize=(5, 4))
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'})
    plt.step(tpr, fpr, color='b', alpha=0.2,
             where='post')
    plt.fill_between(tpr, fpr, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('TPR', size=14)
    plt.ylabel('FPR', size=14)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title, size=15)
    plt.xticks(size=10)
    plt.yticks(size=10)
    
    return None

#     if threshold_selected:
#         p = precision(np.where(threshold == threshold_selected)[0])
#         r = recall(np.where(threshold == threshold_selected)[0])
#         plt.scatter(r, p, marker='*', size=200)
#         plt.vlines(r, ymin=0, ymax=p, linestyles='--')
#         plt.hlines(p, xmin=0, xmax=r, linestyles='--')

#     pr = pd.DataFrame({'precision': precision[:-1], 'recall': recall[:-1],
#                        'threshold': threshold})
#     return pr

def plot_roc_prc(model_results, title='ROC Curve'):
    """Plot an ROC curve for predictions. 
       Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py"""

    # Plot all ROC curves
    plt.figure(figsize=(16, 8))        
    lw = 2    
    
    plt.subplot(1, 2, 1)
    for (model_name, tpr, fpr, auc, _, _, _) in model_results:
        
        plt.plot(fpr, tpr,
                 label='{0} ROC curve (area = {1:0.2f})'
                       ''.format(model_name, auc),
                 linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for generated models', size=15)
    plt.legend(loc="lower right")
    
    plt.subplot(1, 2, 2)
    for (model_name, _, _, _, precision, recall, auc) in model_results:
        
        plt.plot(recall, precision,
                 label='{0} PRC curve (area = {1:0.2f})'
                       ''.format(model_name, auc),
                 linewidth=4)

    plt.plot([0, 1], [0.058, 0.058], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for generated models', size=15)
    plt.legend(loc="lower left")    
    
    plt.show()

    return None

def plot_roc(model_results, title='ROC Curve'):
    """Plot an ROC curve for predictions. 
       Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py"""

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    lw = 2
    
    for (model_name, tpr, fpr, auc) in model_results:
        
        plt.plot(fpr, tpr,
                 label='{0} ROC curve (area = {1:0.2f})'
                       ''.format(model_name, auc),
                 linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve for selected models')
    plt.legend(loc="lower right")
    plt.show()

    return None

def plot_precision_recall(test_y, probs, title='Precision Recall Curve', threshold_selected=None):
    """Plot a precision recall curve for predictions. 
       Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py"""

    precision, recall, threshold = precision_recall_curve(test_y, probs)
    plt.figure(figsize=(5, 4))
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall', size=14)
    plt.ylabel('Precision', size=14)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title, size=15)
    plt.xticks(size=10)
    plt.yticks(size=10)

    if threshold_selected:
        p = precision(np.where(threshold == threshold_selected)[0])
        r = recall(np.where(threshold == threshold_selected)[0])
        plt.scatter(r, p, marker='*', size=200)
        plt.vlines(r, ymin=0, ymax=p, linestyles='--')
        plt.hlines(p, xmin=0, xmax=r, linestyles='--')

    pr = pd.DataFrame({'precision': precision[:-1], 'recall': recall[:-1],
                       'threshold': threshold})
    return pr

def plot_precision_recall(model_results, title='PRC Curve'):
    """Plot an PRC curve for predictions. 
       Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py"""

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    lw = 2
    
    for (model_name, precision, recall, auc) in model_results:
        
        plt.plot(recall, precision,
                 label='{0} PRC curve (area = {1:0.2f})'
                       ''.format(model_name, auc),
                 linewidth=4)

    #plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recal')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for selected models')
    plt.legend(loc="lower right")
    plt.show()

    return None

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.YlOrRd):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.style.use('bmh')
    plt.figure(figsize=(9, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=22)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=20)
    plt.yticks(tick_marks, classes, size=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 size=20)
    plt.grid(None)
    plt.ylabel('True label', size=22)
    plt.xlabel('Predicted label', size=22)
    plt.tight_layout()

def load_train_dataset():
    ## First the dataset indicating whether the customer churned or not.
    train_input   = pd.read_csv('C:/Work/kaggle/wsdm-churn/train.csv', 
                                dtype={'is_churn' : bool, 'msno' : str})

    ## Some basic data about the member 
    members_input = pd.read_csv('C:/Work/kaggle/wsdm-churn/members_v3.csv',
                                dtype={'registered_via' : np.uint8,
                                       'gender' : 'category'})

    train_input = pd.merge(left = train_input, right = members_input, how = 'left', on=['msno'])

    del members_input

    ## Next load in the transactions data
    transactions_input = pd.read_csv('C:/Work/kaggle/wsdm-churn/transactions.csv',
                                     dtype = {'payment_method' : 'category',
                                              'payment_plan_days' : np.uint8,
                                              'plan_list_price' : np.uint8,
                                              'actual_amount_paid': np.uint8,
                                              'is_auto_renew' : np.bool,
                                              'is_cancel' : np.bool})

    transactions_input = pd.merge(left = train_input, right = transactions_input, how='left', on='msno')
    grouped  = transactions_input.copy().groupby('msno')


    shuffle = grouped.agg({'msno' :{'msno_count': 'count'},
                           'plan_list_price' :{'plan_list_price':'sum'},
                           'actual_amount_paid' : {'actual_amount_paid_mean' : 'mean',
                                                   'actual_amount_paid_sum' : 'sum'},
                           'is_cancel' : {'is_cancel_sum': 'sum'}})
    
    shuffle.columns = shuffle.columns.droplevel(0)

    shuffle.reset_index(inplace=True)

    train_input = pd.merge(left = train_input,right = shuffle,how='left',on='msno')

    del transactions_input,shuffle

    return train_input

def load_v2_dataset():
    train_input = pd.read_csv('/home/dissertation/data/train_v2.csv',dtype = {'msno' : str})
    members_input = pd.read_csv('/home/dissertation/data/members_v3.csv',dtype={'registered_via' : np.uint8,
                                                          'gender' : 'category'})

    train_input = pd.merge(left=train_input, right=members_input, how='left', on=['msno'])

    del members_input

    transactions_input = pd.read_csv('/home/dissertation/data/transactions.csv',
                                     dtype = {'payment_method' : 'category',
                                              'payment_plan_days' : np.uint8,
                                              'plan_list_price' : np.uint8,
                                              'actual_amount_paid': np.uint8,
                                              'is_auto_renew' : np.bool,
                                              'is_cancel' : np.bool})

    transactions_input = pd.merge(left = train_input, 
                                  right = transactions_input, 
                                  how='left', 
                                  on='msno')

    grouped  = transactions_input.copy().groupby('msno')

    shuffle = grouped.agg({'msno' : {'total_order' : 'count'},
                           'plan_list_price' : {'plan_net_worth' : 'sum'},
                           'actual_amount_paid' : {'mean_payment_each_transaction' : 'mean',
                                                   'total_actual_payment' : 'sum'},
                           'is_cancel' : {'cancel_times' : lambda x : sum(x==1)}})

    shuffle.columns = shuffle.columns.droplevel(0)
    shuffle.reset_index(inplace=True)
    
    train_input = pd.merge(left = train_input,right = shuffle,how='left',on='msno')
    
    del transactions_input

    return train_input

## https://github.com/Featuretools/predict-customer-churn/blob/master/churn/5.%20Modeling.ipynb

def cost(num_fn, num_tp, num_fp, debug=False):
    num_churns = 13963
    plan_list_price = 150
    plan_incentive_price = 100
    churner_conversion_rate = 0.75
    
    churn_loss = (num_churns * plan_list_price)

    revenue_lost_false_positives = num_fp * (plan_list_price - plan_incentive_price)
    revenue_lost_false_negatives = num_fn * (plan_list_price)
    revenue_retained_true_positives = (num_tp * plan_list_price)

    net_loss = churn_loss - (revenue_retained_true_positives - (revenue_lost_false_positives + revenue_lost_false_negatives))
 
    if debug:
        print("Cost of churns in real life is: ", churn_loss)
        print("This model will identify {0} true churns saving {1:.2f}".format(num_tp, revenue_retained_true_positives))
        print("This model will falsely identify {0} churns losing {1:.2f} in revenue".format(num_fp, revenue_lost_false_positives))
        print("This model will fail to identify {0} churns losing {1:.2f} in revenue".format(num_fn, revenue_lost_false_negatives))
        print("Applying this model will mean a churn loss of {0:.2f}".format(net_loss))
    
    return net_loss

"""
    This function calculates the revenue gain for the given model, represented by each row in the models metrics.
    
    Revenue Lost - (-Gain for successfully re-subscribed customers 
                        + cost of lost revenue for missed churners 
                        + cost of falsely identified churners)
"""
def calc_churn_monetary_value(row, num_members, churn_rate, debug=False):
        
    plan_list_price = 150
    plan_incentive_price = 100
    ntd_eur_conv_rate = 0.029
    churner_conversion_rate = 0.75
    new_customer_acquisition_cost = 3 *  plan_list_price
    
    num_churns = int(num_members * churn_rate)
    monthly_revenue = num_members * plan_list_price
    
    # Find the typical loss of churned customers
    monthly_loss_to_churns = num_churns * (plan_list_price)
    
    #print("Revenue lost to churns is {0:.2f}".format(monthly_loss_to_churns))
    
    # How many of the actual churns did this model predict?
    true_positives = row.tp
    # How many of the actual churns did the model miss?
    false_negatives = row.fn
    # How many of the actual churns did the model predict incorrectly?
    false_positives = row.fp
    # True negatives have zero cost in this context

    ## Lets now calculate the cost of churners based on this model
    
    # 1. FP: If prediction is churn and they did not churn yet are offered the incentive price, then thats lost revenue
    revenue_lost_false_positives = false_positives * (plan_list_price - plan_incentive_price)
    # 2. FN: How much did the churners we missed cost 
    #            Here we have lost revenue, plus need to acquire a new customer to replace them
    revenue_lost_false_negatives = false_negatives * (plan_incentive_price)
    # 3. TP:  Assuming we sucessfully re-subscribe a percentage of the identified churners, what is the retention in revenue
    revenue_retained_true_positives = (true_positives * plan_incentive_price) * churner_conversion_rate
    
    net_loss = monthly_loss_to_churns - (revenue_retained_true_positives - (revenue_lost_false_positives + revenue_lost_false_negatives))
 
    if debug:
        print("Cost of churns in real life is: ", churn_loss)
        print("This model will identify {0} true churns saving {1:.2f}".format(num_tp, revenue_retained_true_positives))
        print("This model will falsely identify {0} churns losing {1:.2f} in revenue".format(num_fp, revenue_lost_false_positives))
        print("This model will fail to identify {0} churns losing {1:.2f} in revenue".format(num_fn, revenue_lost_false_negatives))
        print("Applying this model will mean a churn loss of {0:.2f}".format(net_loss))
       
    """
    
    print("{3} :: FP: {0:.2f}; FN: {1:.2f}; TP: {2:.2f} = {4:.2f}".format(
                    revenue_lost_false_positives, 
                    revenue_lost_false_negatives, 
                    revenue_retained_true_positives,
                    row.classifier,
                    model_churn_cost))
    
    print("{3} :: monthly_loss_to_churns: {0:.2f}; model_churn_cost: {1:.2f}; model_churn_savings: {2:.2f};".format(
                    monthly_loss_to_churns, 
                    model_churn_cost,
                    model_churn_diff,
                    row.classifier))
    """
         
    return net_loss

