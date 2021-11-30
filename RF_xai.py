import shap

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
#from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn import metrics


import numpy as np
from numpyencoder import NumpyEncoder
import time
import json
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

first_time = time.time()


def rf_regression(to_export, cols, all_cols, filename):
    print('************** RF, REGRESSION for dataset %s' %filename)
    #rfc = RandomForestRegressor(criterion= ' squared_error', max_features=int(np.sqrt(len(all_cols))))
    rfc = RandomForestRegressor(max_features=int(np.sqrt(len(all_cols))))
    rfc.fit(to_export['X_train'], to_export['y_train'])

    #prediction_pre = rfc.predict(to_export['X_test_pre'])
    prediction_post = rfc.predict(to_export['X_test_post'])

    print('FEATURE IMPORTANCE')
    for name, importance in zip(rfc.feature_importances_, all_cols):
        print(name, "=", importance)

    features = all_cols
    importances = rfc.feature_importances_
    #print('importances', importances)
    indices = np.argsort(importances)
    plt.figure(figsize=(12, 10))
    plt.title('Random Forest Feature Importances (MDI)')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('images/RF Feature Importances %s' % filename)

    #print('PERM IMPORTANCE REGRESSION 1')
    result = permutation_importance(rfc, to_export['X_test_post'], to_export['y_test_post'], n_repeats=10, random_state=42, n_jobs=2 )
    sorted_idx = result.importances_mean.argsort()
    #print('sorted_idx'), sorted_idx
    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=[all_cols[el] for el in sorted_idx])
    ax.set_title("Permutation Importances (test set)")
    plt.tight_layout()
    plt.savefig('images/' + 'RF Permutation Feature Importance BOXPLOT %s' % filename)

    # Permutation Feature  importance
    #print('PERM IMPORTANCE REGRESSION 2')
    """
    As an alternative, the permutation importances of rf are computed on a held out test set. 
    This shows that the low cardinality categorical feature, sex is the most important feature.

    Also note that both random features have very low importances (close to 0) as expected.
    """
    perm_importance = permutation_importance(rfc, to_export['X_test_post'], to_export['y_test_post'])
    perm_zip = list(zip(all_cols, perm_importance['importances_mean']))
    perm_sorted = sorted(perm_zip, key=lambda x: x[1])
    #print('perm_sorted',perm_sorted)
    plt.figure(figsize=(12, 10))
    x_val = [t[0] for t in perm_sorted]
    y_val = [t[1] for t in perm_sorted]
    plt.barh(x_val, y_val, color='maroon')
    plt.xlabel("Permutation Importance")
    plt.title('RF Permutation Feature Importance')
    plt.tight_layout()
    plt.savefig('images/' + 'RF Permutation Feature Importance %s' % filename)

    #sample_train = shap.sample(to_export['X_train'], nsamples=100, random_state=0)
    # Global explanation for the performance of RANDOM FOREST
    explainer = shap.KernelExplainer(rfc.predict, to_export['X_train'],
                                     nsamples=100,
                                     random_state=90,
                                     link = 'identity',
                                     l1_reg = len(all_cols))

    start_time = time.time()
    shap_values = explainer.shap_values(to_export['X_test_post']) # provo a usare un sample
    #shap_values = explainer.shap_values(shap.sample(to_export['X_test_post'], nsamples=100, random_state=0), l1_reg = len(all_cols))
    #print('shap val regression', shap_values)
    print(f"RF_REGRESSION_SHAP Total time: {(time.time() - start_time) / 60} minutes")
    model_output_rf = (explainer.expected_value + shap_values.sum()).round(4)
    print('model output RF SHAP', model_output_rf)

    # Make plot. Index of [1] arbitrary
    shap.summary_plot(shap_values, to_export['X_test_post'], feature_names=all_cols, plot_type="dot", show=False, figsize=(50,12))
    plt.title('SHAP summary plot - REGRESSION')
    plt.tight_layout()
    plt.savefig('images/RF_regression_Summary_plot_%s' % filename)

    # https://scikit-learn.org/stable/modules/model_evaluation.html#mean-tweedie-deviance # metrics explantion
    diz_rf = {  "shap prediction": model_output_rf,
                'Random Forest feature importance': importances,
                'Mean Absolute Error (MAE):':metrics.mean_absolute_error(to_export['y_test_post'], prediction_post),
                'Mean Squared Error (MSE):': metrics.mean_squared_error(to_export['y_test_post'], prediction_post),
                'Root Mean Squared Error (RMSE):': metrics.mean_squared_error(to_export['y_test_post'], prediction_post, squared=False),
                'Mean Absolute Percentage Error (MAPE):': metrics.mean_absolute_percentage_error(to_export['y_test_post'], prediction_post),
                'Explained Variance Score:': metrics.explained_variance_score(to_export['y_test_post'], prediction_post),
                'Max Error:': metrics.max_error(to_export['y_test_post'], prediction_post),
                'Mean Squared Log Error:': metrics.mean_squared_log_error(to_export['y_test_post'], prediction_post),
                'Median Absolute Error:': metrics.median_absolute_error(to_export['y_test_post'], prediction_post),
                'R^2:': metrics.r2_score(to_export['y_test_post'], prediction_post),
                'Mean Poisson Deviance:': metrics.mean_poisson_deviance(to_export['y_test_post'], prediction_post),
                }

    with open('other_files/RF_METRICS_REGRESSION_POST_%s.json' % filename, 'w', encoding='utf-8') as f1:
        json.dump(diz_rf, f1, cls=NumpyEncoder)
    f1.close()


def rf_classification(to_export, cols, all_cols, filename):
    rfc = RandomForestClassifier(max_features=int(np.sqrt(len(all_cols))))
    """
    RandomForestClassifier : criterion= ' gini',  boostrap = True, oob = True #default args
    """
    rfc.fit(to_export['X_train'], to_export['y_train'])
    class_names = np.unique(to_export['y_train'])

    avg = ''
    if len(class_names) == 2:
        avg += 'binary'
    else:
        avg += 'weighted'

    # F1-score on test set before the drift point
    pred_test_pre = rfc.predict(to_export['X_test_pre'])
    score_test_pre = sklearn.metrics.f1_score(to_export['y_test_pre'], pred_test_pre, average=avg)

    # F1-score on test set after the drift point
    pred_test_post = rfc.predict(to_export['X_test_post'])
    score_test_post = sklearn.metrics.f1_score(to_export['y_test_post'], pred_test_post, average=avg)


    # """
    # The impurity-based feature importance ranks the numerical features to be the most important features.
    # As a result, the non-predictive random_num variable is ranked the most important!
    #
    # This problem stems from two limitations of impurity-based feature importances:
    # - impurity-based importances are biased towards high cardinality features;
    # - impurity-based importances are computed on training set statistics and therefore do not
    # reflect the ability of feature to be useful to make predictions that generalize to the test
    # set (when the model has enough capacity).
    #
    # """

    #print('FEATURE IMPORTANCE ')
    for name, importance in zip(rfc.feature_importances_, all_cols):
        print(name, "=", importance)

    features = all_cols
    importances = rfc.feature_importances_
    indices = np.argsort(importances)
    #print('indices', indices)
    #sorted_zip = sorted(list(zip(importances, all_cols)), key = lambda x:x[0])
    #print('sorted_zip', sorted_zip)
    plt.figure(figsize=(12,10))
    plt.title('Random Forest Feature Importances (MDI)')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('images/RF Feature Importances %s' % filename)


    #print('PERM IMPORTANCE 1')
    result = permutation_importance(rfc, to_export['X_test_post'], to_export['y_test_post'], n_repeats=10, random_state=42, n_jobs=2 )
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=[all_cols[el] for el in sorted_idx])
    ax.set_title("Permutation Importances (test set)")
    plt.tight_layout()
    plt.savefig('images/'+'RF Permutation Feature Importance BOXPLOT %s' % filename)


    # Permutation Feature  importance
    #print('PERM IMPORTANCE 2')
    """
    As an alternative, the permutation importances of rf are computed on a held out test set. 
    This shows that the low cardinality categorical feature, sex is the most important feature.

    Also note that both random features have very low importances (close to 0) as expected.
    """


    perm_importance = permutation_importance(rfc, to_export['X_test_post'], to_export['y_test_post'])
    perm_zip = list(zip(all_cols, perm_importance['importances_mean']))
    perm_sorted = sorted(perm_zip, key= lambda  x:x[1])
    plt.figure(figsize=(12,10))
    x_val = [t[0] for t in perm_sorted]
    y_val = [t[1] for t in perm_sorted]
    plt.barh(x_val, y_val, color='maroon')
    plt.xlabel("Permutation Importance")
    plt.title('RF Permutation Feature Importance')
    plt.tight_layout()
    plt.savefig('images/'+'RF Permutation Feature Importance 2 %s' % filename)



    # Global explanation for the performance of RANDOM FOREST
    #print('faccio explainer')
    sample_train = shap.sample(to_export['X_train'],nsamples=100, random_state=0)
    explainer = shap.KernelExplainer(rfc.predict_proba, sample_train, feature_names=all_cols)
    print('explainer finito CLASSIFIC, ora shap values')
    start_time = time.time()
    shap_values = explainer.shap_values(to_export['X_test_post'])
    #shap_values = explainer.shap_values(shap.sample(to_export['X_test_post'],nsamples=100, random_state=0) ,l1_reg = len(all_cols))

    print(f"RF_CLASSIFICATION_SHAP Total time: {(time.time() - start_time) / 60} minutes")
    print('shap val CLASSIFIC RF', len(shap_values),shap_values)
    model_output = (explainer.expected_value + shap_values[1].sum()).round(4)
    class_pred = np.argmax(abs(model_output))

    # Make plot. Index of [1] arbitrary because 1 = concept drift
    shap.summary_plot(shap_values, to_export['X_test_post'], feature_names=all_cols,plot_type = 'bar', show=False)
    shap.summary_plot(shap_values[1], to_export['X_test_post'], feature_names=all_cols, plot_type = 'dot',show=False)

    plt.title(f'RF SHAP summary plot {filename}')
    plt.tight_layout()
    plt.savefig('images/RF_Summary_plot_%s' % filename)

    #prediction_pre = rfc.predict_proba(to_export['X_test_pre'])
    #pred_pre = rfc.predict(to_export['X_test_pre'])
    #prediction_post = rfc.predict_proba(to_export['X_test_post'])
    #pred_post = rfc.predict(to_export['X_test_post'])


    #print('Compute accuracy metrics for random forest classification')
    diz_rf_cl = {"shap prediction": class_pred,
                "RF train accuracy": rfc.score(to_export['X_train'], to_export['y_train']),
                 'confusion_matrix': metrics.confusion_matrix(to_export['y_test_post'], pred_test_post),
                 "RF test PRE drift accuracy": rfc.score(to_export['X_test_pre'], to_export['y_test_pre']),
                 "Random Forest Classification report PRE drift" : classification_report(to_export['y_test_pre'], pred_test_pre),
                 "RF test POST drift accuracy" : rfc.score(to_export['X_test_post'], to_export['y_test_post']),
                 'Random Forest Classification report POST drift': classification_report(to_export['y_test_post'], pred_test_post),
                 'Random Forest feature importance': importances,
                 'F1_score_pre': score_test_pre,
                 'F1_score_post': score_test_post,
                 'top_k_accuracy': metrics.top_k_accuracy_score(to_export['y_test_post'], pred_test_post, k=2, normalize=False) # Not normalizing gives the number of "correctly" classified samples

              }

    #print('saving file')
    with open('other_files/RF_METRICS_CLASSIFICATION_POST_%s.json' % filename, 'w', encoding='utf-8') as f2:
        json.dump(diz_rf_cl, f2, cls=NumpyEncoder)
    f2.close()

    plt.close('all')


print(f"RF_xai.py Total time: {(time.time() - first_time) / 60} minutes")
print('END RF_xai')
