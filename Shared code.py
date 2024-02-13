# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 01:28:21 2023

@author: liy45
"""


import numpy as np
import matplotlib.pyplot as plt

# import os, os.path
import pandas as pd
import seaborn as sns


# import os, os.path
# import math


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# from sklearn.utils import class_weight

# from sklearn.naive_bayes import ComplementNB, MultinomialNB
# from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
# from keras.callbacks import EarlyStopping


from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, precision_recall_curve, average_precision_score, roc_auc_score, make_scorer

# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer

import xgboost as xgb
import lightgbm as lgb
import shap

from sklearn.inspection import permutation_importance

from collections import Counter
from statsmodels.stats.contingency_tables import cochrans_q 



'''read raw data from here'''


df2 = pd.read_excel(r'your/excel/file/path.xlsx')


def get_lbl(row):
    if row['Nodule_event'] == 1 and row['Nodule_timetoevent'] <= 12*y_followup:
        return 1
    elif row['T_followup'] >= 12*y_followup and (row['Nodule_timetoevent'] > 12*y_followup or row['Nodule_event'] ==0) :
        return 0
    else:
        return np.nan

for y_followup in range(1,6,1):
    df2[f'Ground Truth_{y_followup}'] = df2.apply(get_lbl, axis=1)
    

# df2 = df2.loc[~pd.isna(df2['Ground Truth_3'])].reset_index(drop=True)

log_trans = ['FBS_bsl', 'TG_bsl', 'TCH_bsl', 'LDL_bsl', 'HDL_bsl', 'UA_bsl', 'ALT_bsl', 'AST_bsl', 'GGT_bsl', 'Cr_bsl', 'BMI_bsl', 'MAP_bsl']
for col in log_trans:
    df2[col + '_log'] = np.log10(df2[col])
df2.drop(log_trans, axis=1, inplace=True)

target = ['Ground Truth_3']

unwanted = ['ID_bsl', 'Ht_bsl', 'Wt_bsl', 'SBP_bsl', 'DBP_bsl', 'Nodule_bsl', 'Malignant_bsl', 'T_followup', 
            'Nodule_timetoevent', 'NAFLD_event', 'NAFLD_timetoevent', 'Nodule_event', 'Time']

unwanted.extend([e for e in ['Ground Truth_1', 'Ground Truth_2','Ground Truth_3', 'Ground Truth_4', 'Ground Truth_5'] if e not in target])

df2.drop(unwanted, axis=1, inplace=True)

df2.dropna(axis=0, how = 'any', inplace=True)

df2['Ground Truth_3'].value_counts()
features = [e for e in list(df2.columns) if e not in target]

x = df2.loc[:, features]
y = df2.loc[:, target]




'''check correlation'''

def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', cmap="YlGnBu",
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70}
                )
    plt.show()
    
correlation_heatmap(x)
# correlation_heatmap(df2)


def create_descriptive_table(x):
    
    first_item_list, second_item_list = [], []
    feature_num_list, feature_std_list = [], []
    
    feature_list = list(x.columns)
    
    for feature in feature_list:
        
        a = Counter(x[feature])
        
        if len(a) >= 10: # conintuous:
            first_item_list.append(feature)
            second_item_list.append(feature)
            feature_num_list.append(x[feature].mean())
            feature_std_list.append(x[feature].std())            
        else:
            total_n = sum(a.values())
            for e in list(a.keys()):
                first_item_list.append(feature)
                second_item_list.append(e)
                feature_num_list.append(a[e])
                feature_std_list.append(a[e]/total_n)
    
    df = pd.DataFrame(list(zip(first_item_list, second_item_list, feature_num_list, feature_std_list)),
                      columns=['Features', 'Class', 'Count or mean', '% or SD']
                      )
    
    return df

stat_x = create_descriptive_table(x)



''' 2. Hyperparameter tuning'''

def param_tune_single(ml_model = 'RF', test_param = 'n_estimators', param_range = None):
       
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 3214, stratify=y)
    
    train_results = []
    test_results = []
    param_grid = {test_param: param_range}
    
    for param in param_grid[test_param]:
        
        dict_temp = {test_param: param}
        if ml_model == 'RF':
            clf = RandomForestClassifier(random_state = 10, n_jobs=-1, class_weight = 'balanced', criterion='gini', **dict_temp)

        elif ml_model == 'SVM':
            clf = SVC(probability=True, class_weight='balanced', max_iter = 100000, kernel = 'rbf', gamma = 'scale', **dict_temp)
     
        elif ml_model == 'KNN':
            clf = KNeighborsRegressor(weights = 'distance', metric = 'minkowski',  **dict_temp)
              
        elif ml_model == 'XGB':
            clf = xgb.XGBClassifier(subsample=0.5, random_state = 20, n_jobs = -1, eta = 0.1, gamma = 0.1, objective = 'binary:logistic', **dict_temp) # around 70
    
        elif ml_model == 'LGB':
            clf = lgb.LGBMClassifier(subsample=0.5, objective  = 'binary', random_state = 20, n_jobs = -1, learning_rate = 0.1,
                                     verbose = 1, **dict_temp) # around 70   
        else:
            clf = LogisticRegression(penalty='l2', class_weight = 'balanced', max_iter = 100000, **dict_temp, solver = 'liblinear') #,
        
        print('training {} at {}...'.format(test_param, param))
        clf.fit(x_train.values, np.ravel(y_train.values))
        train_pred = clf.predict(x_train.values)
        fpr, tpr, thresholds = roc_curve(np.ravel(y_train), train_pred)
        roc_auc = auc(fpr, tpr)
        train_results.append(roc_auc)
        y_pred = clf.predict(x_test.values)
        fpr, tpr, thresholds = roc_curve(np.ravel(y_test), y_pred)
        roc_auc = auc(fpr, tpr)
        test_results.append(roc_auc)
    
    line1, = plt.plot(param_grid[test_param], train_results, 'b', label='Train AUC')
    line2, = plt.plot(param_grid[test_param], test_results, 'r', label='Test AUC')
    plt.legend()
    plt.ylabel('AUC score')
    plt.xlabel(test_param)
    plt.show()
    
param_tune_single(ml_model = 'LR', test_param = 'C', param_range = list(map(lambda x:pow(2, x), list(range(3, 10, 1)))))



''' define classifier and grid search target'''

def select_clf(ml_model = 'SVM'):
   
    if ml_model == 'SVM':
        clf = SVC(probability=True, class_weight='balanced', max_iter = 100000, kernel = 'rbf', gamma = 'scale')
        ''' probability: This must be enabled prior to calling fit, will slow down that method as it internally uses 5-fold cross-validation, 
            and predict_proba may be inconsistent with predict'''
        param_grid = {#'kernel': ['linear', 'poly', 'rbf'], #'sigmoid'
                      'C': list(map(lambda x:pow(2, x), list(range(2, 6, 1)))),
                      # 'gamma': ['scale', 'auto']
                      }
        
    elif ml_model == 'RF':
        clf = RandomForestClassifier(random_state = 10, n_jobs=-1, class_weight = 'balanced', criterion='gini')
        param_grid = {# 'bootstrap': [True],
                      'max_depth': list(range(4, 8, 1)),
                      'max_features': list(range(10,16,1)),
                      'min_samples_leaf': list(range(60, 160, 20)),
                      'min_samples_split': list(range(10, 200, 30)),
                      'n_estimators': [4, 8, 16, 32],
                      # 'class_weight' : ['balanced', 'balanced_subsample'],
                      # 'criterion': ['gini', 'entropy']
                      }
        
    elif ml_model == 'KNN':
        clf = KNeighborsRegressor(weights = 'distance', metric = 'minkowski')
        param_grid = {#'weights': ['uniform', 'distance'],
                      'n_neighbors': list(range(40, 80, 5)),
                      'leaf_size' : list(range(15, 26, 5)),
                      # 'metric': ['euclidean', 'manhattan', 'minkowski']
                      }   
        
    elif ml_model == 'XGB':
        clf = xgb.XGBClassifier(subsample=0.5, random_state = 20, n_jobs = -1, gamma = 0.1, eta = 0.1, objective = 'binary:logistic') # around 70
        param_grid = {#'eta' : [0.1, 0.2, 0.3, 0.4, 0.5] ,
                      'max_depth' : list(range(2, 11, 2)),
                      'min_child_weight' : list(range(10, 36, 5)),
                      # 'gamma' : [ 0.0, 0.1, 0.2, 0.3],
                      'colsample_bytree' : list(np.linspace(0.5, 0.7, 5, endpoint=True)),
                      'n_estimators':  [8, 16, 32, 64, 128]
                      }
        
    elif ml_model == 'LGB':
        clf = lgb.LGBMClassifier(subsample=0.5, objective  = 'binary', random_state = 20, n_jobs = -1, learning_rate = 0.1,
                                 verbose = 1) 
        param_grid = {#'learning_rate' : [0.1, 0.2, 0.3, 0.4, 0.5],
                      'num_leaves' : list(range(5, 11, 2)),
                      'min_child_samples': list(range(100, 200, 20)),
                      #'gamma' : [0.0, 0.1, 0.2, 0.3],
                      'colsample_bytree' : list(np.linspace(0.2, 0.8, 4, endpoint=True)),
                      'n_estimators':  [8, 16, 32, 64]
                      }
    else:
        clf = LogisticRegression(class_weight = 'balanced', max_iter = 1000000, solver = 'liblinear')
        param_grid = {
                    # 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                      'C': list(map(lambda x:pow(2, x), list(range(4, 8, 1))))
                      }

    return clf, param_grid





def grid_search_wrapper(x_dev, y_dev, clf, scoring = 'PR'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    if scoring == 'PR':
        
        score_func = 'average_precision'
    
    elif scoring == 'ROC':
        
        score_func = 'roc_auc'
             
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20)
    
    grid_search = GridSearchCV(clf, param_grid, scoring = score_func, refit = False,
                           cv=skf, return_train_score=False, n_jobs=-1, verbose = 2)
    
    grid_search.fit(x_dev, np.ravel(y_dev))

    print('Best params for {}'.format(scoring))
    print(grid_search.best_params_)


    return grid_search


def train_test_model(x_train, y_train, x_val, ml_model):

    ml_model.fit(x_train, np.ravel(y_train))
    
    pred_prob = ml_model.predict_proba(x_val)
    
    return pred_prob


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]




''' execution of step 2: to get the best hyperparameters via GridSearchCV on the whole dataset'''

# x_dev, x_test, y_dev, y_test = scale_dataset(x, y, scale = 'None')

clf, param_grid = select_clf('LGB')
grid_search_result = grid_search_wrapper(x, y, clf, scoring = 'ROC') # tune hyperparameters to maximize scoring
print(grid_search_result.best_params_)
# grid_search_clf = grid_search_result.best_estimator_  
    


''' Best hyperparameters:
    Random Forest: {'max_depth': 6, 'max_features': 12, 'min_samples_leaf': 60, 'min_samples_split': 10, 'n_estimators': 32}
    KNN : {'leaf_size': 15, 'n_neighbors': 40}
    SVM : {'C': 32}
    Logistic regression: {'C': 16}
    XGBoost: {'colsample_bytree': 0.6, 'max_depth': 4, 'min_child_weight': 10, 'n_estimators': 32}
    LightGBM: {'colsample_bytree': 0.8, 'min_child_samples': 160, 'n_estimators': 64, 'num_leaves': 9}
'''



''' Step 3. cross-validation with the best hyperparameters 
    get the 95% threshold'''

def train_clf(x_dev, y_dev, ml_model = 'SVM'):
    
    
    if ml_model == 'SVM':
        clf = SVC(probability=True, class_weight='balanced', max_iter = 100000, kernel = 'rbf', gamma = 'scale',
                  C = 32)
        # x_dev, x_test, y_dev, y_test = split_scale (x, y, scale = 'Standard')
        
    elif ml_model == 'RF':
        clf = RandomForestClassifier(random_state = 10, n_jobs=-1, class_weight = 'balanced', criterion='gini', 
                                     max_depth = 6, max_features = 12, min_samples_leaf = 60, min_samples_split = 10,
                                     n_estimators = 32)
        # x_dev, x_test, y_dev, y_test = split_scale (x, y, scale = 'None')

    elif ml_model == 'KNN':
        clf = KNeighborsRegressor(weights = 'distance', metric = 'minkowski', 
                                  leaf_size = 15, n_neighbors = 40)
        # x_dev, x_test, y_dev, y_test = split_scale (x, y, scale = 'None')
        
    
    else:
        clf = LogisticRegression(class_weight = 'balanced', max_iter = 1000000, solver = 'liblinear', 
                                 C = 16)
        # x_dev, x_test, y_dev, y_test = split_scale (x, y, scale = 'None')

    x_train,x_val,y_train,y_val = train_test_split(x_dev, y_dev,test_size = 0.2, random_state = 10, shuffle = True, stratify = y_dev)    

    clf.fit(x_train, np.ravel(y_train))
    
    if ml_model == 'KNN':
        pred_score = clf.predict(x_val)
    else:
        pred_score = clf.predict_proba(x_val)[:,1]
      
    fpr, tpr, auc_thresholds = roc_curve(y_val, pred_score)
    
    balanced_idx = np.argmax(tpr - fpr)
    balanced_threshold = auc_thresholds[balanced_idx]     

    return clf, balanced_threshold


    
def get_metrics_testset(y_true, y_scores, b_threshold):
        
    fpr, tpr, auc_thresholds = roc_curve(y_true, y_scores)
    
    roc_score = auc(fpr, tpr)
    
    ap = average_precision_score(y_true, y_scores)
               
    y_pred_lbl = adjusted_classes(y_scores, b_threshold)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_lbl).ravel()
    
    accuracy = (tp+tn)/len(y_pred_lbl)

    tpr = tp / (tp+fn)
    tnr = tn / (tn+fp)
                   
    pre = tp / (tp+fp)

    print('\nAt threshold {:.4f}\n'.format(b_threshold))
    print(pd.DataFrame(np.array([tp,fp,fn,tn]).reshape(2,2),
                       columns=['pos', 'neg'], 
                       index=['pred_pos', 'pred_neg']))
    
    print('\nAccuracy: {:.4f}\nSensitivity: {:.4f}\nSpecificity: {:.4f}\nPrecision: {:.4f}\nROC: {:.4f}\nAP: {:.4f}'.format(
            accuracy, tpr, tnr, pre, roc_score, ap))

    return tn, fp, fn, tp, accuracy, tpr, tnr, pre, roc_score, ap



''' Step 4. To get the permutation importance score'''

def plot_feature_importance(df, top = 5, title_text = None):
    
    if top is not None:   
        df_fea = df.sort_values('Mean').iloc[-top:]
    else:
        df_fea = df.sort_values('Mean')
         
    ax = df_fea.plot.barh(y='Mean', xerr = 'SEM',  color='#86bf91')
    
    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Switch off ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    # Draw vertical axis lines
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)
        
    ax.legend().set_visible(False)
    
    # Set x-axis label
    ax.set_xlabel("Feature importance score", labelpad=20, weight='bold', size=18)
    
    # Set y-axis label
    ax.set_ylabel("Feautures", labelpad=20, weight='bold', size=16)
    
    ax.set_title(title_text + ' - ' + ml_type, size = 20)
    
    


''' execution of step 3 and 4'''

ml_type = 'LR'

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 12345)

thresh_list, acc_list, tpr_list, tnr_list, pre_list, auc_list, ap_list= [], [], [], [], [], [], []
tn_list, fp_list, fn_list, tp_list = [], [], [] ,[]
fold_list = []
df_dict = {}
feature_importance_list = []
fold_num = 1
for train_index, test_index in skf.split(x, y):
    
    x_dev, x_test = x.iloc[train_index], x.iloc[test_index]
    y_dev, y_test = y.iloc[train_index], y.iloc[test_index]
    
    clf, op_threshold = train_clf(x_dev.values, y_dev.values, ml_model = ml_type)
    
    if ml_type == 'KNN':
        y_test_score = clf.predict(x_test.values)
    else:
        y_test_score = clf.predict_proba(x_test.values)[:,1]
    
    y_pred_lbl = [1 if y >= op_threshold else 0 for y in y_test_score]
    
    df_dict[fold_num] = pd.DataFrame(list(zip(test_index, y_test_score, y_pred_lbl, y_test.values.flatten())),
                         columns = ['ID', 'Pred score', 'Pred lbl', 'Truth'])
 
    tn, fp, fn, tp, accuracy, tpr, tnr, pre, roc_score, ap_score = get_metrics_testset(y_test.to_numpy(), y_test_score, op_threshold)
    
    fold_list.append(fold_num)
    thresh_list.append(op_threshold)
    acc_list.append(accuracy)
    tpr_list.append(tpr)
    tnr_list.append(tnr)
    pre_list.append(pre)
    auc_list.append(roc_score)
    ap_list.append(ap_score)
    tn_list.append(tn)
    fp_list.append(fp)
    fn_list.append(fn)
    tp_list.append(tp)
    
    feature_importance = permutation_importance(clf, x_test.values, y_test.values, scoring='roc_auc', n_repeats=50, random_state=10, n_jobs=-1) # scoring='neg_log_loss'

    result = pd.DataFrame(data = list(zip(fold_list, thresh_list, tn_list, fp_list, fn_list, tp_list, 
                                          acc_list, tpr_list, tnr_list, pre_list, auc_list, ap_list)),
                          columns= ['Fold', 'Threshold', 'TN', 'FP', 'FN', 'TP',
                                    'Accuracy', 'Recall', 'Specificity', 'Precision', 'ROC', 'AP'])

    feature_importance_list.append(feature_importance.importances_mean)
    
    fold_num +=1

df_result = pd.concat(df_dict).droplevel(1, axis=0).reset_index()  
df_result.rename(columns={'index': 'Fold'}, inplace=True) 
    
a = np.vstack(feature_importance_list)

df_importance = pd.DataFrame(list(zip(np.mean(a, axis = 0), np.std(a, axis = 0)/np.sqrt(len(a)))),
                             index=list(x_test.columns),
                             columns = ['Mean', 'SEM'])

df_result.to_excel(r'C:\Users\liy45\Desktop\{}.xlsx'.format(ml_type), sheet_name='case', index = False)
with pd.ExcelWriter(r'C:\Users\liy45\Desktop\{}.xlsx'.format(ml_type), engine='openpyxl', mode='a') as writer:  
    # df_result.to_excel(writer, sheet_name='case', index = False)
    result.to_excel(writer, sheet_name='sum', index=False)
    df_importance.to_excel(writer, sheet_name='importance', index=True)  

plot_feature_importance(df_importance, top = 10, title_text = target[0].upper())




''' new ml models: XGBoost, Light GBM
XGBoost: {'colsample_bytree': 0.6, 'max_depth': 4, 'min_child_weight': 10, 'n_estimators': 32}
LightGBM: {'colsample_bytree': 0.8, 'min_child_samples': 160, 'n_estimators': 64, 'num_leaves': 9}
'''

# dtrain = xgb.DMatrix(x_train, label=y_train)
# dval = xgb.DMatrix(x_val, label=y_val)
# dtest = xgb.DMatrix(x_test, label=y_test)

# param  = {'colsample_bytree': 0.3, 'eta': 0.1, 'gamma': 0.2, 'max_depth': 5, 'min_child_weight': 7}
# evallist = [(dval, 'eval'), (dtrain, 'train')]

def train_clf_new(x_dev, y_dev, ml_model = 'XGB'):

    # x_dev, x_test, y_dev, y_test = split_scale (x, y, scale = 'None')
    x_train,x_val,y_train,y_val = train_test_split(x_dev,y_dev,test_size = 0.2, random_state = 10, shuffle = True, stratify = y_dev)    

    if ml_model == 'XGB':
        clf = xgb.XGBClassifier(subsample=0.5, random_state = 20, n_jobs = -1, gamma = 0.1, eta = 0.1, objective = 'binary:logistic',
                                colsample_bytree= 0.6, max_depth=4, min_child_weight=10, n_estimators = 32)
                            
        clf.fit(x_train, np.ravel(y_train), 
                eval_set = [(x_train, np.ravel(y_train)), (x_val, np.ravel(y_val))], 
                early_stopping_rounds = 5,
                eval_metric='logloss') #48 rounds
    
        pred_score = clf.predict_proba(x_val, ntree_limit=clf.best_ntree_limit)[:,1]
        
        # y_test_score = clf.predict_proba(x_test, ntree_limit=clf.best_ntree_limit)[:,1]
    
        # xgb.plot_importance(clf)
        
    else: 
        clf = lgb.LGBMClassifier(subsample=0.5, objective  = 'binary', random_state = 20, n_jobs = -1, learning_rate = 0.1,
                                 verbose = 1,colsample_bytree = 0.8, min_child_samples = 160, n_estimators=64, num_leaves=9)
                                 
   
        clf.fit(x_train, np.ravel(y_train), 
                eval_set = [(x_train, np.ravel(y_train)), (x_val, np.ravel(y_val))], 
                early_stopping_rounds = 5,
                eval_metric='logloss') #48 rounds
    
        pred_score = clf.predict_proba(x_val)[:,1]
        
        # y_test_score = clf.predict_proba(x_test)[:,1]
        
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(x_test)

    shap.summary_plot(shap_values, x_test, plot_type="bar")    
    
      
    fpr, tpr, auc_thresholds = roc_curve(y_val, pred_score)
    
    balanced_idx = np.argmax(tpr - fpr)
    balanced_threshold = auc_thresholds[balanced_idx] 
    
    return clf, balanced_threshold


ml_type = 'XGB'

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 12345)

thresh_list, acc_list, tpr_list, tnr_list, pre_list, auc_list, ap_list= [], [], [], [], [], [], []
tn_list, fp_list, fn_list, tp_list = [], [], [] ,[]
fold_list = []
df_dict = {}
feature_importance_list = []

fold_num = 1
for train_index, test_index in skf.split(x, y):
    
    x_dev, x_test = x.iloc[train_index], x.iloc[test_index]
    y_dev, y_test = y.iloc[train_index], y.iloc[test_index]
    
    clf, op_threshold = train_clf_new(x_dev.reset_index(drop = True), y_dev.reset_index(drop = True), ml_model = ml_type)
    y_test_score = clf.predict_proba(x_test)[:,1]
    y_pred_lbl = [1 if y >= op_threshold else 0 for y in y_test_score]
    df_dict[fold_num] = pd.DataFrame(list(zip(test_index, y_test_score, y_pred_lbl, y_test.values.flatten())),
                         columns = ['ID', 'Pred score', 'Pred lbl', 'Truth'])
    tn, fp, fn, tp, accuracy, tpr, tnr, pre, roc_score, ap_score = get_metrics_testset(y_test.to_numpy(), y_test_score, op_threshold)
    
    fold_list.append(fold_num)
    thresh_list.append(op_threshold)
    acc_list.append(accuracy)
    tpr_list.append(tpr)
    tnr_list.append(tnr)
    pre_list.append(pre)
    auc_list.append(roc_score)
    ap_list.append(ap_score)
    tn_list.append(tn)
    fp_list.append(fp)
    fn_list.append(fn)
    tp_list.append(tp)

    feature_importance = permutation_importance(clf, x_test, y_test, scoring='roc_auc', n_repeats=50, random_state=10, n_jobs = -1) #scoring='neg_log_loss',

    result = pd.DataFrame(data = list(zip(fold_list, thresh_list, tn_list, fp_list, fn_list, tp_list, 
                                          acc_list, tpr_list, tnr_list, pre_list, auc_list, ap_list)),
                          columns= ['Fold', 'Threshold', 'TN', 'FP', 'FN', 'TP',
                                    'Accuracy', 'Recall', 'Specificity', 'Precision', 'ROC', 'AP'])

    feature_importance_list.append(feature_importance.importances_mean)
    
    fold_num +=1
    
df_result = pd.concat(df_dict).droplevel(1, axis=0).reset_index()  
df_result.rename(columns={'index': 'Fold'}, inplace=True)     
    
    
a = np.vstack(feature_importance_list)

df_importance = pd.DataFrame(list(zip(np.mean(a, axis = 0), np.std(a, axis = 0)/np.sqrt(len(a)))),
                             index=list(x_test.columns),
                             columns = ['Mean', 'SEM'])

df_result.to_excel(r'C:\Users\liy45\Desktop\{}.xlsx'.format(ml_type), sheet_name='case', index = False)
with pd.ExcelWriter(r'C:\Users\liy45\Desktop\{}.xlsx'.format(ml_type), engine='openpyxl', mode='a') as writer:  
    # df_result.to_excel(writer, sheet_name='case', index = False)
    result.to_excel(writer, sheet_name='sum', index=False)
    df_importance.to_excel(writer, sheet_name='importance', index=True) 

plot_feature_importance(df_importance, top = 10, title_text = target[0].upper())




'''Significance: Cochran's Q test for identical binomial proportions'''


''' all non-sig except for RF preop even better'''

ml_type = 'RF'

df_ml = pd.read_excel(r'your\result\path_{}.xlsx'.format(ml_type), 
                         sheet_name = 'case', header = 0)

df_bsl = pd.read_excel(r'your\result\path_LR.xlsx', 
                         sheet_name = 'case', header = 0)
# array_like, 2d (N, k)
df_final = df_ml.merge(df_bsl, on =['ID','Fold'] , suffixes=('_ml', '_bsl'), how = 'outer')

df_final['sucess_ml'] = 1 - abs(df_final['Pred lbl_ml'] - df_final['Truth_ml'])
df_final['sucess_bsl'] = 1 - abs(df_final['Pred lbl_bsl'] - df_final['Truth_bsl'])

print('ML: {:.2%}'.format(len(df_final.loc[df_final['sucess_ml'] == 1])/len(df_final)))
print('Baseline: {:.2%}'.format(len(df_final.loc[df_final['sucess_bsl'] == 1])/len(df_final)))
res = cochrans_q(df_final[['sucess_ml', 'sucess_bsl']])
print(res)


''' only positive'''
df_final = df_final.loc[df_final['Truth_ml'] == 1].reset_index(drop=True)
print('ML: {:.2%}'.format(len(df_final.loc[df_final['sucess_ml'] == 1])/len(df_final)))
print('Baseline: {:.2%}'.format(len(df_final.loc[df_final['sucess_bsl'] == 1])/len(df_final)))
res = cochrans_q(df_final[['sucess_ml', 'sucess_bsl']])
print(res)
