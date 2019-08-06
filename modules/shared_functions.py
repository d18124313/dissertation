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
import autosklearn.classification

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, f1_score, log_loss, precision_recall_curve, average_precision_score, precision_recall_curve, recall_score, precision_score, average_precision_score, balanced_accuracy_score

import numpy as np
import pandas as pd
import time

plt.style.use('seaborn')


METRIC_COLS = ["label", "classifier", "sampling_method", "tn", "fn", "tp", "fp", "accuracy", "precision", "recall", "neg_recall",                          "f1_score", "log_loss", "train_time", "cv_time", "aucroc", "auprc", "balanced_accuracy", "cv_score_mean", "cv_score_std"]
    
class DummySampler:

    def sample(self, X, y):
        return X, y

    def fit(self, X, y):
        return self

    def fit_resample(self, X, y):
        return self.sample(X, y)

def OHE(df, col = ['city','registered_via']):
   
    if col:        
        print('Categorical columns in dataset', col)

        c2,c3 = [],{}
        for c in col:
            if df[c].nunique()>2 :
                c2.append(c)
                c3[c] = 'ohe_'+c

        df = pd.get_dummies(df, columns=c2, drop_first=True, prefix=c3)
    else:
        print('No categorical columns provided for OHE')
        
    return df

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

def normalise(train_set, test_set):
    print("Applying normalisation to train and test sets")
    scale_columns = ['bd', 'age_cat', 'total_actual_payment', 'mean_payment_each_transaction', \
                     'plan_net_worth', 'total_order', 'cancel_times']  
 
    scale_us = intersection(train_set.columns, scale_columns) 
    
    scaler = preprocessing.StandardScaler().fit(train_set[scale_us])

    train_set[scale_us] = scaler.transform(train_set[scale_us])
    test_set[scale_us] = scaler.transform(test_set[scale_us])
    
    return train_set, test_set

def prepare_train_test_data(X_train, X_test, y_train, y_test, sampler = DummySampler(), cat_col = ['city','registered_via']): 
    
    X_train_meta = X_train.head(1)
    X_test_meta = X_test.head(1)

    print("PRE-SAMPLING:", X_train.shape, y_train.shape, Counter(y_train))
    ## This will fit the provided sampling approach to the training data 
    X_train, y_train = sampler.fit_resample(X_train, y_train)
    print("POST-SAMPLING:", X_train.shape, y_train.shape, Counter(y_train))
               
    # Reconstruct the dataframes
    X_train = pd.DataFrame(X_train, columns=X_train_meta.columns)
    X_test = pd.DataFrame(X_test, columns=X_test_meta.columns)
    y_train = pd.DataFrame(y_train, columns=['is_churn'])
    y_test = pd.DataFrame(y_test, columns=['is_churn'])
    
    # The dtypes are cast to object during sampling, need to set them back  
    print("Set the train df types correctly based on the test set")
    for col, dtype in zip(X_test_meta.columns, X_test_meta.dtypes):
        #print("Set:", col, "as", dtype)
        X_train[col] = X_train[col].astype(dtype)
    
    # Normalise the dataset
    X_train, X_test = normalise(X_train, X_test)
    
    # One-hot-encode the categorical features
    X_train = OHE(X_train, cat_col)
    X_test = OHE(X_test, cat_col)
    
    print("X_train: ", X_train.shape, y_train.shape)
    print("X_test: ", X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test

def perform_experiment(X_train, X_test, y_train, y_test, classifier_set, sampler, iterations = 1, 
                       cv_iter = None, cat_col = ['city','registered_via'], feat_defs = None):
    
    metrics_all = pd.DataFrame()
    model_results = list()

    for i in range(0, iterations):

        print("Model Build Iteration", i)       
            
        if i == 0:
            X_train, X_test, y_train, y_test = \
                        prepare_train_test_data(X_train, X_test, y_train, y_test,
                                                sampler = sampler[1],
                                                cat_col = cat_col)

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
                                          cv_iter = cv_iter,
                                          feat_defs = feat_defs
                                         )

        metrics = model_build_results[0]
        metrics['sample'] = i
        metrics_all = metrics_all.append(metrics)

        if i == 0:
            plot_data = []
            for res in model_build_results[1]:
                label, model_name, sampling_method, model, fpr, tpr, roc_auc, precision, recall, prc_auc, feat_importance = res
                plot_data.append((model_name, tpr, fpr, roc_auc, precision, recall, prc_auc))  
                model_results.append((label, model_name, sampling_method, feat_importance, tpr, fpr, roc_auc, precision, recall, prc_auc))

            plot_roc_prc(plot_data)

    return (metrics_all, model_results)

def clean_dataset(df):
    return df.dropna().copy()

def train_model(X_train, X_test, y_train, y_test, classifiers, 
                sampling_method = 'None', 
                cv_iter=None,
                feat_defs=None):
    
    target_index = 0
    results = []
    
    metrics = pd.DataFrame(columns=METRIC_COLS)

    for name, model, params, metric in classifiers:
        print('Training {0} classifier'.format(name))
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
            # Special case for Autosklearn to inform it of the feature types
            if name == 'ASKLEARN' and feat_defs:
                model.fit(X_train.drop(columns=['registration_init_time', 'registration_init_time_dt'], axis=1, errors='ignore'), 
                          y_train,
                          metric = metric,
                          feat_type = feat_defs)
            else:
                model.fit(X_train.drop(columns=['registration_init_time', 'registration_init_time_dt'], axis=1, errors='ignore'), 
                          y_train)      
            
        print('Generating test scores for {0} classifier'.format(name))
        y_predict = model.predict(X_test.drop(columns=['registration_init_time', 'registration_init_time_dt'], axis=1, errors='ignore'))
        y_predict_prob = model.predict_proba(X_test.drop(columns=['registration_init_time', 'registration_init_time_dt'], axis=1, errors='ignore'))[:,1]
        
        finish_train = time.time()
        
        # Compute ROC/AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_predict_prob)
        roc_auc = auc(fpr, tpr)     
        
        # Compute Precision-Recall
        precision, recall, thresholds = precision_recall_curve(y_test, y_predict_prob)
        # calculate precision-recall AUC
        prc_auc = auc(recall, precision)    

        ## Lets keep track of CV processing time also
        finish_cv = finish_train
        
        cv_score = []
        if 'ASKLEARN' in name or 'TPOT' in name:
            cv_metric = None
        else:
            cv_metric = metric
            
        if cv_iter:   
            print("Performing {}-fold CV on test set using {} metric".format(cv_iter, metric))
            cv_score = cross_val_score(model,
                                       X_test.drop(columns=['registration_init_time', 'registration_init_time_dt'], axis=1, errors='ignore'), 
                                       y_test, 
                                       scoring = cv_metric, 
                                       n_jobs = -1,
                                       cv=cv_iter)    
            finish_cv = time.time()
        
            
        metrics = metrics.append(
                generate_eval_metrics(name, model.__class__.__name__, 
                                      sampling_method, 
                                      y_test,
                                      y_predict, 
                                      y_predict_prob,
                                      cv_score, 
                                      finish_train - start, 
                                      finish_cv - finish_train))
        
        feat_imp = None
        if hasattr(model, 'feature_importances_'):
            feat_imp = collect_feature_importances(model, X_test.columns)
        
        results.append((name, model.__class__.__name__, sampling_method, model, fpr, tpr, roc_auc, precision, recall, prc_auc, feat_imp))
        
    ## Compute the monetary cost of the trained churn models
    metrics['model_churn_cost'] = metrics.apply(calc_churn_monetary_value, 
                                                axis = 1, 
                                                args = [len(y_test), np.mean(y_test)])    

    return (metrics, results)

## Print the feature importance 

def collect_feature_importances(model, df_columns):
    
    feature_index = np.flip(np.argsort(model.feature_importances_), axis=0)
    column_names = df_columns
    ordered_features = list()
    feat_importances = list()

    for i in feature_index:
        feat_importances.append(np.round(model.feature_importances_[i], 3))
        ordered_features.append(column_names[i])
        
    importances = pd.DataFrame({'feat_importances': feat_importances, 'ordered_features': ordered_features})
    return importances

def generate_eval_metrics(label, class_name, sampling_method, y_test, y_predict, y_predict_prob, cv_score, train_time, cv_time):

    fpr, tpr, _ = roc_curve(y_test, y_predict_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_predict_prob)
    
    conf_m = confusion_matrix(y_test, y_predict)
    tn = conf_m[0][0]
    fn = conf_m[1][0]
    tp = conf_m[1][1]
    fp = conf_m[0][1]
    
    mean_score = None
    std_score = None
    if cv_score is not None and len(cv_score) > 0:
        mean_score =  np.mean(cv_score)
        std_score =  np.std(cv_score)        

    metric_entry = pd.DataFrame([[label, class_name, 
                                  sampling_method,
                                  tn, fn, tp, fp,
                                  accuracy_score(y_test, y_predict),
                                  precision_score(y_test, y_predict),
                                  recall_score(y_test, y_predict),
                                  tn / (tn + fp),
                                  f1_score(y_test, y_predict), 
                                  log_loss(y_test, y_predict),
                                  train_time,
                                  cv_time,
                                  auc(fpr, tpr), 
                                  auc(rec, prec),
                                  balanced_accuracy_score(y_test, y_predict),
                                  mean_score if mean_score else -1, 
                                  std_score if std_score else -1]], 
                                columns=METRIC_COLS)       

    return metric_entry

def filter_top_model_results(top_models, all_model_results):
    plot_data = list()
    for idx, row in top_models.iterrows():
        for res in all_model_results:
            run_results = res[2]        
            for alg_results in run_results:
                if alg_results[0] == row.label and alg_results[1] == row.classifier and alg_results[2] == row.sampling_method:
                    label, model_name, sampling_method, _, tpr, fpr, roc_auc, precision, recall, prc_auc = alg_results

                    plot_data.append(("{} {}".format(model_name, sampling_method), tpr, fpr, roc_auc, precision, recall, prc_auc))
    return plot_data
    
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

## https://github.com/Featuretools/predict-customer-churn/blob/master/churn/5.%20Modeling.ipynb


def calc_cost(num_tp, num_tn, num_fp, num_fn, debug=False):
    
    ## Cost*FN + 0*TN + Incentive*FP + Incentive*TP
    
    # Incentive cost is 25 TWD off subscription for 4 months
    plan_incentive_cost = (25 * 4) 
    # CAC
    customer_acq_cost = 500

    cost = customer_acq_cost*num_fn + plan_incentive_cost*num_fp + plan_incentive_cost*num_tp
    
    if debug:
        print("Worst case cost (We ignore churn and customers leave): ", customer_acq_cost * (num_tp + num_fn))
        print("Best case cost (We identify all churners and offer the incentive to keep them): ", plan_incentive_cost * (num_tp + num_fn))
        print("Offering all customers the incentive would cost", plan_incentive_cost * sum([num_tp, num_tn, num_fp, num_fn]))
        print("Applying this model will mean a churn cost of {0:.2f}".format(cost))
    
    return cost

def loss(num_fn, num_tp, num_fp, debug=False):
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
        
    ntd_eur_conv_rate = 0.029
    churner_conversion_rate = 0.75
    
    #print("Revenue lost to churns is {0:.2f}".format(monthly_loss_to_churns))
    
    # How many of the actual churns did this model predict?
    true_positives = row.tp
    # How many of the actual churns did the model miss?
    false_negatives = row.fn
    # How many of the actual churns did the model predict incorrectly?
    false_positives = row.fp
    # True negatives have zero cost in this context
    true_negatives = row.tn

    ## Lets now calculate the cost of churners based on this model
    
    cost = calc_cost(true_positives, true_negatives, false_positives, false_negatives, debug)
         
    return cost

def store(results, file_name):    
    """
        (sampling_approach_metrics_df, (modelling_results)) =>
            modelling_results => (label, model_name, feat_importance, tpr, fpr, roc_auc, precision, recall, prc_auc)

        e.g. data[0][1][1][2] returns the feature importance dataframe
    """

    # Write (overwrite) the file to store the experiment results
    with open(file_name, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        print("Writing results to", f.name)
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
        
def log(msg, file_name):    
    # Write (overwrite) the file to store the experiment results
    with open(file_name, 'a') as f:
        f.write(msg + '\n')
