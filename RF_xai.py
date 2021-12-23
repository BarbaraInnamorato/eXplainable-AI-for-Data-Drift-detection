import pandas as pd
import shap

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


import numpy as np
from numpyencoder import NumpyEncoder
import time
import json
import matplotlib
import matplotlib.pyplot as plt

import Perm_importance

matplotlib.use('Agg')

first_time = time.time()

"""
def rf_regression(to_export,  all_cols, filename):
    print('************** RF, REGRESSION for dataset %s' %filename)
    rfc = RandomForestRegressor(max_features=int(np.sqrt(len(all_cols))))
    rfc.fit(to_export['X_train'], to_export['y_train'])

    prediction_post = rfc.predict(to_export['X_test_post'])

    print('FEATURE IMPORTANCE')
    for name, importance in zip(rfc.feature_importances_, all_cols):
        print(name, "=", importance)

    features = all_cols
    importances = rfc.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(12, 10))
    plt.title('Random Forest Feature Importances (MDI) - Regression')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(f'images/RF Feature Importances {filename}')




    perm_imp = Perm_importance.compute_pfi(rfc, to_export, all_cols, filename)
"""


def rf_classification(to_export, all_cols, filename):
    """
    RandomForestClassifier : criterion= ' gini',  boostrap = True, oob = True #default args
    """

    rfc = None
    if filename != 'anas':

        #rfc = RandomForestClassifier(max_features=int(np.sqrt(len(all_cols))))
        rfc = RandomForestClassifier(max_features='sqrt', oob_score=True)
        rfc.fit(to_export['X_train'], to_export['y_train'])
        class_names = np.unique(to_export['y_train'])
        print('class_names', class_names)

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
        print('pred_test_post',pred_test_post)
        score_test_post = sklearn.metrics.f1_score(to_export['y_test_post'], pred_test_post, average=avg)

        # Global explanation for the performance of RANDOM FOREST
        sample_train = shap.sample(to_export['X_train'], nsamples=10, random_state=90)  # nsamples=100
        explainer = shap.KernelExplainer(rfc.predict_proba,
                                         sample_train,
                                         feature_names=all_cols,
                                         link='identity',
                                         l1_reg=len(all_cols))

        print('explainer finito CLASSIFIC, ora shap values')
        start_time = time.time()
        shap_values = explainer.shap_values(to_export['X_test_post'])
        tot_time = (time.time() - start_time) / 60
        print('shap val CLASSIFIC RF', len(shap_values), shap_values)
        model_output = (explainer.expected_value + shap_values[1].sum()).round(4)
        class_pred = np.argmax(abs(model_output))
        print('class_pred',class_pred)
        fig = plt.figure(constrained_layout=True)

        print(f'PLOTTING SHAP BAR {filename}')
        shap.summary_plot(shap_values, to_export['X_test_post'], feature_names=all_cols, plot_type='bar',show=False)
        plt.title(f'RF SHAP summary plot {filename}')
        fig.tight_layout()
        fig.savefig(f'images/RF_BAR_Summary_plot_{filename}')


        shap.summary_plot(shap_values[class_pred], to_export['X_test_post'], feature_names=all_cols, plot_type='dot',show=False)


        diz_rf_cl = {"shap prediction": class_pred,
                     "RF train accuracy": rfc.score(to_export['X_train'], to_export['y_train']),
                     'confusion_matrix': metrics.confusion_matrix(to_export['y_test_post'], pred_test_post),
                     "RF test PRE drift accuracy": rfc.score(to_export['X_test_pre'], to_export['y_test_pre']),
                    # "Random Forest Classification report PRE drift": classification_report(to_export['y_test_pre'],pred_test_pre),
                     "RF test POST drift accuracy": rfc.score(to_export['X_test_post'], to_export['y_test_post']),
                     'Random Forest Classification report POST drift': classification_report(to_export['y_test_post'],pred_test_post),
                    # 'Random Forest feature importance': importances,
                     'F1_score_pre': score_test_pre,
                     'F1_score_post': score_test_post,
                     'time': tot_time
                     }

        with open('other_files/RF_METRICS_CLASSIFICATION_POST_%s.json' % filename, 'w', encoding='utf-8') as f2:
            json.dump(diz_rf_cl, f2, cls=NumpyEncoder)
        f2.close()


    else:
        rfc = RandomForestRegressor(max_features='sqrt', oob_score=True)
        rfc.fit(to_export['X_train'], to_export['y_train'])

        prediction_post = rfc.predict(to_export['X_test_post'])

        sample_train = shap.sample(to_export['X_train'], nsamples=100, random_state=90)

        # Global explanation for the performance of RANDOM FOREST
        explainer = shap.KernelExplainer(rfc.predict,
                                         sample_train,
                                         feature_names=all_cols,
                                         link='identity',
                                         l1_reg=len(all_cols)
                                         )

        print('explainer finito REGRESSION, ora shap values')
        start_time = time.time()
        shap_values = explainer.shap_values(to_export['X_test_post'])  # provo a usare un sample
        print('shap val RF regression \n', shap_values)
        end_time = (time.time() - start_time) / 60
        model_output_rf = (explainer.expected_value + shap_values.sum()).round(4)
        print('model output RF SHAP', model_output_rf)

        # Make plot
        shap.summary_plot(shap_values, to_export['X_test_post'], feature_names=all_cols, plot_type="dot", show=False, figsize=(50, 12))
        plt.title('SHAP summary plot - REGRESSION')
        plt.tight_layout()
        plt.savefig('images/RF_regression_Summary_plot_%s' % filename)

        # https://scikit-learn.org/stable/modules/model_evaluation.html#mean-tweedie-deviance # metrics explantion
        diz_rf = {"shap prediction": model_output_rf,
                  "oob_score": rfc.oob_score_,
                  'Mean Absolute Error (MAE):': metrics.mean_absolute_error(to_export['y_test_post'], prediction_post),
                  'Mean Squared Error (MSE):': metrics.mean_squared_error(to_export['y_test_post'], prediction_post),
                  'Root Mean Squared Error (RMSE):': metrics.mean_squared_error(to_export['y_test_post'], prediction_post, squared=False),
                  'Mean Absolute Percentage Error (MAPE):': metrics.mean_absolute_percentage_error(to_export['y_test_post'], prediction_post),
                  'Explained Variance Score:': metrics.explained_variance_score(to_export['y_test_post'], prediction_post),
                  'Max Error:': metrics.max_error(to_export['y_test_post'], prediction_post),
                  'Mean Squared Log Error:': metrics.mean_squared_log_error(to_export['y_test_post'], prediction_post),
                  'Median Absolute Error:': metrics.median_absolute_error(to_export['y_test_post'], prediction_post),
                  'R^2:': metrics.r2_score(to_export['y_test_post'], prediction_post),
                  'Mean Poisson Deviance:': metrics.mean_poisson_deviance(to_export['y_test_post'], prediction_post),
                  'time': end_time

                  }
        with open('other_files/RF_METRICS_REGRESSION_POST_%s.json' % filename, 'w', encoding='utf-8') as f1:
            json.dump(diz_rf, f1, cls=NumpyEncoder)
        f1.close()

    #print('FEATURE IMPORTANCE ')
    for name, importance in zip(rfc.feature_importances_, all_cols):
        print(name, "=", importance)

    features = all_cols
    importances = rfc.feature_importances_
    print('importances', importances.argsort())
    indices = np.argsort(importances)
    print('indices', indices)
    plt.figure(figsize=(12,10))
    plt.title('Random Forest Feature Importances (MDI)- Classification')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(f'images/RF Feature Importances {filename}')

    #save_importances = pd.DataFrame(importances, columns='value')
    #save_importances.to_excel(f'other_files/importances_{filename}')

    plt.close('all')

    Perm_importance.compute_pfi(rfc, to_export, all_cols, filename)




print(f"RF_xai.py Total time: {(time.time() - first_time) / 60} minutes")
print('END RF_xai')
