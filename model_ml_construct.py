import pandas as pd
import os
import time

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, \
    roc_auc_score, recall_score, matthews_corrcoef, confusion_matrix, precision_recall_curve, auc

from joblib import dump, load, Parallel, delayed 
from sklearn.base import clone

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def split_rus(df):
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=6)  
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=6)  
    
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_val, X_test, y_train_resampled, y_val, y_test

def split_ros(df):
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=6)  
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=6)  
    
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_val, X_test, y_train_resampled, y_val, y_test

def split_external(df):
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    return X, y

lgb_params = {'random_state': [42],
              'objective': ['binary'],
              'boosting_type': ['gbdt'],
              'scale_pos_weight': [1, 1.2, 1.4, 1.6, 1.8, 2],
              'num_leaves': [i for i in range(31, 80, 16)],
              'n_estimators': [i for i in range(100, 501, 100)]}

RF_params = {'random_state': [42],
             'max_depth': [i for i in range(1, 10, 2)],
             'criterion': ['gini'],
             'class_weight': ['balanced', 'balanced_subsample'],
             'n_estimators': [i for i in range(10, 101, 10)]}

SVM_params = {'random_state': [42],
              'kernel': ['rbf'],
              'probability': [True],
              'class_weight': [None, 'balanced'],
              'C': [i * 0.1 for i in range(5, 51, 5)],
              'gamma': ['scale', 'auto', 1e-2, 5e-2, 1e-1, 5e-1]}

xgb_params = {'random_state': [42],
              'booster': ['gbtree'],
              'objective': ['binary:logistic'],
              'max_depth': [i for i in range(1, 10, 2)],
              'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
              'n_estimators': [i for i in range(10, 101, 10)]}

def pr_auc_score(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return auc(recall, precision)

dict_metrics = {
    'SE': make_scorer(recall_score),
    'SP': make_scorer(recall_score, pos_label=0),
    'ACC': make_scorer(accuracy_score),
    'BA': make_scorer(balanced_accuracy_score),
    'MCC': make_scorer(matthews_corrcoef),
    'AUC': make_scorer(roc_auc_score, needs_proba=True),  
    'PR-AUC': make_scorer(pr_auc_score, needs_proba=True)}

model1 = LGBMClassifier(verbose=-1, n_jobs=1)
model2 = RandomForestClassifier(n_jobs=1)
model3 = SVC(probability=True, verbose=False)  
model4 = XGBClassifier(verbosity=0, n_jobs=1)

def train_and_evaluate(param, model, X_train, y_train, X_val, y_val):
    m = clone(model)
    m.set_params(**param)
    m.fit(X_train, y_train)
    y_pred = m.predict(X_val)
    y_proba = m.predict_proba(X_val)[:, 1] if hasattr(m, "predict_proba") else None
    scores = {name: scorer._score_func(y_val, y_proba if scorer._kwargs.get('needs_proba') else y_pred, **scorer._kwargs)
              for name, scorer in dict_metrics.items()}
    return {'params': param, 'scores': scores, 'model': m}

def grid_search(model, params, X_train, y_train, X_val, y_val, n_jobs=-1):
    param_list = list(ParameterGrid(params))
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_and_evaluate)(param, model, X_train, y_train, X_val, y_val)
        for param in param_list
    )
    best_result = None
    best_score = -1
    for res in results:
        mcc = res['scores']['MCC']
        if mcc > best_score:
            best_score = mcc
            best_result = res
    for res in results:
        res['is_best'] = (res == best_result)
    results_df = pd.DataFrame([{'is_best': res['is_best'], **res['params'], **res['scores']} for res in results])
    best_model = best_result['model']
    return best_model, results_df

def grid_cvs(X_train, X_val, y_train, y_val, data_type, sampling_method, rep, model, n_jobs=-1):
    if model == 'lgb':
        best_model, results_df = grid_search(model1, lgb_params, X_train, y_train, X_val, y_val, n_jobs=n_jobs)
        results_df.to_excel(f'./ml_gs_results/{data_type}/{sampling_method}/lgb+{rep}.xlsx', index=False)
        dump(best_model, f'./ml_final_models/{data_type}/{sampling_method}/lgb+{rep}.joblib')

    elif model == 'RF':
        best_model, results_df = grid_search(model2, RF_params, X_train, y_train, X_val, y_val, n_jobs=n_jobs)
        results_df.to_excel(f'./ml_gs_results/{data_type}/{sampling_method}/RF+{rep}.xlsx', index=False)
        dump(best_model, f'./ml_final_models/{data_type}/{sampling_method}/RF+{rep}.joblib')

    elif model == 'SVM':
        best_model, results_df = grid_search(model3, SVM_params, X_train, y_train, X_val, y_val, n_jobs=n_jobs)
        results_df.to_excel(f'./ml_gs_results/{data_type}/{sampling_method}/SVM+{rep}.xlsx', index=False)
        dump(best_model, f'./ml_final_models/{data_type}/{sampling_method}/SVM+{rep}.joblib')

    elif model == 'xgb':
        best_model, results_df = grid_search(model4, xgb_params, X_train, y_train, X_val, y_val, n_jobs=n_jobs)
        results_df.to_excel(f'./ml_gs_results/{data_type}/{sampling_method}/xgb+{rep}.xlsx', index=False)
        dump(best_model, f'./ml_final_models/{data_type}/{sampling_method}/xgb+{rep}.joblib')


def test(X_test, y_test, data_type, sampling_method, rep, model):
    model_to_test = load(f'./ml_final_models/{data_type}/{sampling_method}/{model}+{rep}.joblib')  
    y_pred = model_to_test.predict(X_test) 
    y_proba = model_to_test.predict_proba(X_test)[:, 1]

    def new_confusion_matrix(y_true, y_pred):
        return confusion_matrix(y_true, y_pred, labels=[0, 1])

    def se(y_true, y_pred):
        cm = new_confusion_matrix(y_true, y_pred)
        return cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])

    def sp(y_true, y_pred):
        cm = new_confusion_matrix(y_true, y_pred)
        return cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])

    metrics = {
        'SE': se(y_test, y_pred),
        'SP': sp(y_test, y_pred),
        'ACC': accuracy_score(y_test, y_pred),
        'BA': balanced_accuracy_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba),
        'PR-AUC': pr_auc_score(y_test, y_proba)
    }
    return metrics

start_time = time.time()
data_types = ['binder', 'agonist', 'antagonist']

for data_type in data_types:
    for sampling_method in ['rus', 'ros']:
        if not os.path.isdir(f'ml_gs_results/{data_type}/{sampling_method}'):
            os.makedirs(f'ml_gs_results/{data_type}/{sampling_method}')
        if not os.path.isdir(f'ml_final_models/{data_type}/{sampling_method}'):
            os.makedirs(f'ml_final_models/{data_type}/{sampling_method}')
        if not os.path.isdir(f'ml_results_all/{data_type}/{sampling_method}'):
            os.makedirs(f'ml_results_all/{data_type}/{sampling_method}')

        df_result_training = pd.DataFrame([])
        df_result_test = pd.DataFrame(columns=['model', 'rep', 'SE', 'SP', 'ACC', 'BA', 'MCC', 'AUC', 'PR-AUC'])

        models = ['lgb', 'RF', 'SVM', 'xgb']
        reps = ['descriptors', 'maccs', 'morgan', 'rdk', 'mol2vec']
        for model in models:
            for rep in reps:
                df = pd.read_csv(f"datasets/{data_type}_train/{rep}/{rep}.csv")
                if sampling_method == 'rus':
                    X_train, X_val, X_test, y_train, y_val, y_test = split_rus(df)
                else:
                    X_train, X_val, X_test, y_train, y_val, y_test = split_ros(df)

                grid_cvs(X_train, X_val, y_train, y_val, data_type, sampling_method, rep, model, n_jobs=1)
                df_cv = pd.read_excel(f'./ml_gs_results/{data_type}/{sampling_method}/{model}_+{rep}.xlsx')

                best_row = df_cv[df_cv['is_best'] == True].copy() 
                best_row['model'] = model
                best_row['rep'] = rep
                df_result_training = pd.concat([df_result_training, best_row], ignore_index=True)

                metrics_test = test(X_test, y_test, data_type, sampling_method, rep, model)

                numbers = [round(metrics_test[metric], 3) for metric in ['SE', 'SP', 'ACC', 'BA', 'MCC', 'AUC', 'PR-AUC']]
                df_result_test = pd.concat([df_result_test, pd.DataFrame([[model, rep] + numbers], columns=df_result_test.columns)], ignore_index=True)

        df_training = pd.DataFrame(columns=['model', 'rep', 'SE', 'SP', 'ACC', 'BA', 'MCC', 'AUC', 'PR-AUC'])
        for metric in df_training.columns[2:]:  
            df_training[metric] = df_result_training[metric].round(3).map(str)
        df_training['model'] = df_result_training['model']  
        df_training['rep'] = df_result_training['rep']

        df_training.to_excel(f'ml_results_all/{data_type}/{sampling_method}/results.xlsx', sheet_name='training', index=False)

        with pd.ExcelWriter(f'ml_results_all/{data_type}/{sampling_method}/results.xlsx', engine='openpyxl', mode='a') as writer:
            df_result_test.to_excel(writer, sheet_name='test', index=False)

        df_result_external = pd.DataFrame(columns=['model', 'rep', 'SE', 'SP', 'ACC', 'BA', 'MCC', 'AUC', 'PR-AUC'])
        for model in models:
            for rep in reps:
                df = pd.read_csv(f"dataset/{data_type}_external/{rep}/{rep}.csv")
                X_test, y_test = split_external(df)
                metrics_external = test(X_test, y_test, data_type, sampling_method, rep, model)
                numbers = [round(metrics_external[metric], 3) for metric in ['SE', 'SP', 'ACC', 'BA', 'MCC', 'AUC', 'PR-AUC']]
                df_result_external = pd.concat([df_result_external, pd.DataFrame([[model, rep] + numbers], columns=df_result_external.columns)], ignore_index=True)

        with pd.ExcelWriter(f'ml_results_all/{data_type}/{sampling_method}/results.xlsx', engine='openpyxl', mode='a') as writer:
            df_result_external.to_excel(writer, sheet_name='external', index=False)

end_time = time.time()
elapsed_time = end_time - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)
print(f"Run time: {hours} h/{minutes} min/{seconds} s")
