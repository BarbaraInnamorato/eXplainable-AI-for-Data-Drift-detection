import warnings
warnings.filterwarnings("ignore")

import Perm_importance

import shap

import sklearn
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import numpy as np
from numpyencoder import NumpyEncoder
import time
import json

import matplotlib.pyplot as plt

from collections import OrderedDict

def plot_oob(to_export, all_cols, filename):
    print('PLOT OOB')
    ensemble_clfs = [
        ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(
                                warm_start=True,
                                oob_score=True,
                                max_features="sqrt",
                                random_state=123,
        ))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    # Range of `n_estimators` values to explore.
    min_estimators = 15
    max_estimators = 175

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(to_export['X_train'], to_export['y_train'])

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    plt.figure()
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_trees")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.savefig(f'oob_error_{filename}')


    # random forest FEATURE IMPORTANCE
    for name, importance in zip(clf.feature_importances_, all_cols):
        print(name, "=", importance)

    features = all_cols
    importances = clf.feature_importances_
    print('importances', importances.argsort())
    indices = np.argsort(importances)
    print('indices', indices)
    plt.figure()
    plt.title('Random Forest Feature Importances (MDI)- Classification')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(f'images/RF Feature Importances {filename}')

    class_names = np.unique(to_export['y_train'])
    print('class_names', class_names)

    avg = ''
    if len(class_names) == 2:
        avg += 'binary'
    else:
        avg += 'weighted'

    # F1-score on test set before the drift point
    pred_test_pre = clf.predict(to_export['X_test_pre'])
    score_test_pre = sklearn.metrics.f1_score(to_export['y_test_pre'], pred_test_pre, average=avg)

    # F1-score on test set after the drift point
    pred_test_post = clf.predict(to_export['X_test_post'])
    print('pred_test_post', pred_test_post)
    score_test_post = sklearn.metrics.f1_score(to_export['y_test_post'], pred_test_post, average=avg)

    # Global explanation for the performance of RANDOM FOREST
    sample_train = shap.sample(to_export['X_train'], nsamples=10, random_state=90)  # nsamples=100
    explainer = shap.KernelExplainer(clf.predict_proba,
                                     sample_train,
                                     #to_export['X_train'],
                                     feature_names=all_cols,
                                     link='identity',
                                     l1_reg=len(all_cols))

    # explainer = shap.TreeExplainer(clf,
    #                              to_export['X_train'],
    #                              feature_names=all_cols,
    #                              )

    print('explainer finito CLASSIFIC, ora shap values')
    start_time = time.time()
    shap_values = explainer.shap_values(to_export['X_test_post'])
    tot_time = (time.time() - start_time) / 60
    print('shap val CLASSIFIC RF', len(shap_values), shap_values)
    model_output = (explainer.expected_value + shap_values[1].sum()).round(4)
    class_pred = np.argmax(abs(model_output))
    print('class_pred', class_pred)
    fig = plt.figure(constrained_layout=True)

    print(f'PLOTTING SHAP BAR {filename}')
    shap.summary_plot(shap_values, to_export['X_test_post'], feature_names=all_cols, plot_type='bar',
                      show=False)
    plt.title(f'RF SHAP summary plot {filename}')
    fig.tight_layout()
    fig.savefig(f'images/RF_BAR_Summary_plot_{filename}')

    shap.summary_plot(shap_values[0], to_export['X_test_post'], type='dot', show=False)

    diz_rf_cl = {"shap prediction": class_pred,
                 "RF train accuracy": clf.score(to_export['X_train'], to_export['y_train']),
                 'confusion_matrix': metrics.confusion_matrix(to_export['y_test_post'], pred_test_post),
                 "RF test PRE drift accuracy": clf.score(to_export['X_test_pre'], to_export['y_test_pre']),
                 # "Random Forest Classification report PRE drift": classification_report(to_export['y_test_pre'],pred_test_pre),
                 "RF test POST drift accuracy": clf.score(to_export['X_test_post'], to_export['y_test_post']),
                 'Random Forest Classification report POST drift': classification_report(
                     to_export['y_test_post'], pred_test_post),
                 'Random Forest feature importance': importances,
                 'F1_score_pre': score_test_pre,
                 'F1_score_post': score_test_post,
                 'time': tot_time
                 }

    with open('other_files/RF_METRICS_CLASSIFICATION_POST_%s.json' % filename, 'w', encoding='utf-8') as f2:
        json.dump(diz_rf_cl, f2, cls=NumpyEncoder)
    f2.close()
    Perm_importance.compute_pfi(clf, to_export, all_cols, filename)



def plot_oob_regression(to_export, all_cols, filename):
    print('PLOT OOB')
    ensemble_clfs = [
        (   "RandomForestRegressor, max_features='sqrt'",
             RandomForestRegressor(
                 warm_start=True,
                 oob_score=True,
                 max_features="sqrt",
                 random_state=123)
            )]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    # Range of `n_estimators` values to explore.
    min_estimators = 15
    max_estimators = 175

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(to_export['X_train'], to_export['y_train'])

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_trees")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.savefig(f'oob_error_{filename}')

    # random forest FEATURE IMPORTANCE
    for name, importance in zip(clf.feature_importances_, all_cols):
        print(name, "=", importance)

    features = all_cols
    importances = clf.feature_importances_
    print('importances', importances.argsort())
    indices = np.argsort(importances)
    print('indices', indices)
    # plt.figure(figsize=(12, 10))
    plt.title('Random Forest Feature Importances (MDI)- Classification')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(f'images/RF Feature Importances {filename}')

    prediction_post = clf.predict(to_export['X_test_post'])

    sample_train = shap.sample(to_export['X_train'], nsamples=100, random_state=90)

    # Global explanation for the performance of RANDOM FOREST
    explainer = shap.KernelExplainer(clf.predict,
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

    # Make shap plot
    plt.figure()
    shap.summary_plot(shap_values, to_export['X_test_post'], feature_names=all_cols, plot_type="dot",
                      show=False, figsize=(50, 12))
    plt.title('SHAP summary plot - REGRESSION')
    plt.tight_layout()
    plt.savefig('images/RF_regression_Summary_plot_%s' % filename)

    # https://scikit-learn.org/stable/modules/model_evaluation.html#mean-tweedie-deviance # metrics explantion
    diz_rf = {"shap prediction": model_output_rf,
              "oob_score": clf.oob_score_,
              'Random Forest feature importance': importances,
              'Mean Absolute Error (MAE):': metrics.mean_absolute_error(to_export['y_test_post'],
                                                                        prediction_post),
              'Mean Squared Error (MSE):': metrics.mean_squared_error(to_export['y_test_post'],
                                                                      prediction_post),
              'Root Mean Squared Error (RMSE):': metrics.mean_squared_error(to_export['y_test_post'],
                                                                            prediction_post, squared=False),
              'Mean Absolute Percentage Error (MAPE):': metrics.mean_absolute_percentage_error(
                  to_export['y_test_post'], prediction_post),
              'Explained Variance Score:': metrics.explained_variance_score(to_export['y_test_post'],
                                                                            prediction_post),
              'Max Error:': metrics.max_error(to_export['y_test_post'], prediction_post),
              'Mean Squared Log Error:': metrics.mean_squared_log_error(to_export['y_test_post'],
                                                                        prediction_post),
              'Median Absolute Error:': metrics.median_absolute_error(to_export['y_test_post'],
                                                                      prediction_post),
              'R^2:': metrics.r2_score(to_export['y_test_post'], prediction_post),
              'Mean Poisson Deviance:': metrics.mean_poisson_deviance(to_export['y_test_post'],
                                                                      prediction_post),
              'time': end_time

              }
    with open('other_files/RF_METRICS_REGRESSION_POST_%s.json' % filename, 'w', encoding='utf-8') as f1:
        json.dump(diz_rf, f1, cls=NumpyEncoder)
    f1.close()

    Perm_importance.compute_pfi(clf, to_export, all_cols, filename)
