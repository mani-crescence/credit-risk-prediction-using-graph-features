import numpy as np
# from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, precision_score, roc_auc_score, roc_curve, auc
from sklearn import metrics
import shap
import pandas as pd
# from scipy.special import xlogy
import pickle
import os
from shap.plots import colors
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter


def build_shap_plot(model, model_name, X_train, X_test, directory, discretization_type=None):
    num_features = len(X_test.columns)
    expected_max_evals = 2 * num_features + 1
    if expected_max_evals < 256:
        max_evals = 256
    else:
      max_evals = 2 * num_features + 1
      
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_test, max_evals=max_evals)
    shap_values_array = shap_values.values

    mean_abs_shap_values = np.mean(np.abs(shap_values.values))

    exit(mean_abs_shap_values)

    shap_df = pd.DataFrame(
        shap_values_array,
        columns = X_test.columns)

    if discretization_type == 'None':
        files_dir = directory+ 'loan'
        os.makedirs(files_dir, exist_ok=True)
        shap_df.to_csv(files_dir + '/shap_data_' + model_name.lower() + '.csv')
    else:
        files_dir = directory +  discretization_type.lower()
        os.makedirs(files_dir, exist_ok=True)
        print(files_dir + '/shap_data_bip_mod_' + model_name.lower() + '.csv')
        shap_df.to_csv(files_dir + '/shap_data_bip_mod_' + model_name.lower() + '.csv')

def evaluation(model, model_name, X_tr, y_tr, X_te, y_te, classic_result = None):
    model.fit(X_tr,y_tr)
    y_pred = model.predict(X_te)
    
    y_probs = model.predict_proba(X_te)[:, 1]
    
    # print(y_probs)
    # exit()
    
    
    accuracy = metrics.accuracy_score(y_te,y_pred)
    f1 = metrics.f1_score(y_te,y_pred)
    rappel = metrics.recall_score(y_te, y_pred)
    precision = metrics.precision_score(y_te, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_te, y_probs)
    auc = metrics.auc(fpr, tpr)
    roc_auc = metrics.roc_auc_score(y_te, y_probs)

    if classic_result is None:
        return {
        'acc': "{:.4f}".format(accuracy),
        'f1': "{:.4f}".format(f1),
        'rappel' : "{:.4f}".format(rappel),
        'precision': "{:.4f}".format(rappel),
        'auc': "{:.4f}".format(auc),
        'roc_auc' : "{:.4f}".format(roc_auc),
        }, {},  model
    else:
        classic_acc = float(classic_result["BASELINE"][model_name]['acc'])
        classic_f1 = float(classic_result["BASELINE"][model_name]['f1'])
        classic_rappel = float(classic_result["BASELINE"][model_name]['rappel'])
        classic_precision = float(classic_result["BASELINE"][model_name]['precision'])
        classic_auc = float(classic_result["BASELINE"][model_name]['auc'])
        classic_roc_auc = float(classic_result["BASELINE"][model_name]['roc_auc'])

        if classic_acc == 0:
            percent_acc = 0
        else:
            percent_acc = ((float(accuracy) - classic_acc)/classic_acc)*100

        if classic_f1 == 0:
            percent_f1 = 0
        else:
            percent_f1 = ((float(f1) - classic_f1)/classic_f1)*100
            
        if classic_rappel == 0:
            percent_rappel = 0
        else:
            percent_rappel = ((float(accuracy) - classic_rappel)/classic_rappel)*100

        if classic_precision == 0:
            percent_precision = 0
        else:
            percent_precision = ((float(precision) - classic_precision)/classic_precision)*100    
            
        if classic_auc == 0:
            percent_auc= 0
        else:
            percent_auc = ((float(auc) - classic_auc)/classic_auc)*100

        if classic_roc_auc == 0:
            percent_roc_auc = 0
        else:
            percent_roc_auc = ((float(roc_auc) - classic_roc_auc)/classic_roc_auc)*100    
    

        return  {
        'acc': "{:.4f}".format(accuracy),
        'f1': "{:.4f}".format(f1),
        'rap': "{:.4f}".format(rappel),
        'pre': "{:.4f}".format(precision),
        'auc': "{:.4f}".format(auc),
        'roc': "{:.4f}".format(roc_auc)
        }, {
        'acc': "{:.3f}".format(percent_acc),
        'f1': "{:.3f}".format(percent_f1),
        'rap': "{:.4f}".format(percent_rappel),
        'pre': "{:.4f}".format(percent_precision),
        'auc': "{:.4f}".format(percent_auc),
        'roc': "{:.4f}".format(percent_roc_auc)
        }, model

def train(models,train, test, target, classic_result = None):
    
    X_train = train.drop(target,axis=1)
    y_train = train[target]
    X_test = test.drop(target, axis=1)
    y_test = test[target]
    metrics_real = {}
    metrics_with_percent = {}
   
   
    for key, value in models.items():
        if not classic_result:
            classic_result = None
        real, per, _ = evaluation(value, key, X_train, y_train, X_test, y_test, classic_result)
        metrics_real[key] = real
        metrics_with_percent[key] = per


    return metrics_real, metrics_with_percent

        
        
        
        
        
        
        
        
        
    