import warnings
warnings.filterwarnings("ignore")
import shap
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import recall_score
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
    clf = None
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

    plt.title(f'OOB_ERROR {filename}')
    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_trees")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.savefig(f'oob_error_{filename}')

    # random forest FEATURE IMPORTANCE
    for name, importance in zip(clf.feature_importances_, all_cols):
        print(name, "=", importance)

    importances = clf.feature_importances_
    ord_zip = list(zip(all_cols, importances))
    sort_ord_zip = sorted(ord_zip, key=lambda x:x[1], reverse=True)
    indices = np.argsort(importances)

    plt.figure()
    plt.title('Random Forest Feature Importances (MDI)- Classification')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [all_cols[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(f'images/RF Feature Importances {filename}')

    class_names = np.unique(to_export['y_train'])

    avg = ''
    if len(class_names) == 2:
        avg += 'binary'
    else:
        avg += 'weighted'

    # F1-score on test set before the drift point
    pred_test_pre = clf.predict(to_export['X_test_pre'])
    score_test_pre = f1_score(to_export['y_test_pre'], pred_test_pre, average=avg)

    # F1-score on test set after the drift point
    pred_test_post = clf.predict(to_export['X_test_post'])
    score_test_post = f1_score(to_export['y_test_post'], pred_test_post, average=avg)

    # Global explanation for the performance of RANDOM FOREST
    sample_train = shap.sample(to_export['X_train'], nsamples=10, random_state=90)  # nsamples=100
    explainer = shap.KernelExplainer(clf.predict_proba,
                                     sample_train,
                                     feature_names=all_cols,
                                     link='identity',
                                     l1_reg=len(all_cols)
                                     )

    print('----Computing SHAP values')
    start_time = time.time()
    shap_values = explainer.shap_values(to_export['X_test_post'])
    tot_time = (time.time() - start_time) / 60

    fig = plt.figure(constrained_layout=True)
    shap.summary_plot(shap_values, to_export['X_test_post'], feature_names=all_cols, plot_type='bar', show=False)
    plt.title(f'RF SHAP summary plot {filename}')
    fig.tight_layout()
    fig.savefig(f'images/RF_BAR_Summary_plot_{filename}')

    conf_matrix = metrics.confusion_matrix(to_export['y_test_post'], pred_test_post)
    print(f'confusion matrix POST {filename}')
    print(conf_matrix)

    class_rep_post_drift = classification_report(to_export['y_test_post'], pred_test_post)
    print(f'class_rep_post_drift {filename}')
    print(class_rep_post_drift)

    fpr, tpr, thresholds = metrics.roc_curve(to_export['y_test_post'], pred_test_post, pos_label=2)

    diz_rf_cl = {'oob_score': clf.oob_score_,
                 'Random Forest feature importance': sort_ord_zip,

                 'Random Forest Classification report POST drift': class_rep_post_drift,
                 'confusion_matrix': conf_matrix,

                 'RF test PRE drift accuracy': accuracy_score(to_export['y_test_pre'], pred_test_pre),
                 'Precision_pre': precision_score(to_export['y_test_pre'], pred_test_pre, average=avg),
                 'Recall_pre': recall_score(to_export['y_test_pre'], pred_test_pre, average=avg),
                 'F1_score_pre': score_test_pre,

                 'RF test POST drift accuracy': accuracy_score(to_export['y_test_post'], pred_test_post),
                 'Precision_post': precision_score(to_export['y_test_post'], pred_test_post, average=avg),
                 'Recall_post': recall_score(to_export['y_test_post'], pred_test_post, average=avg),
                 'F1_score_post': score_test_post,

                 'AUC': metrics.auc(fpr, tpr),
                 'time': tot_time
                 }

    with open('other_files/' +f'RF_METRICS_CLASSIFICATION_POST_{filename}.json', 'w', encoding='utf-8') as f2:
        json.dump(diz_rf_cl, f2, cls=NumpyEncoder)
    f2.close()

    return clf


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
    plt.figure()
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.title(f'OOB_ERROR {filename}')
    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_trees")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.savefig(f'oob_error_{filename}')

    # random forest FEATURE IMPORTANCE
    for name, importance in zip(clf.feature_importances_, all_cols):
        print(name, "=", importance)

    importances = clf.feature_importances_
    ord_zip = list(zip(all_cols, importances))
    sort_ord_zip = sorted(ord_zip, key=lambda x: x[1])
    indices = np.argsort(importances)

    plt.figure()
    plt.title('Random Forest Feature Importances (MDI)- Regression')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [all_cols[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(f'images/RF_Feature_Importances_{filename}')

    prediction_post = clf.predict(to_export['X_test_post'])
    prediction_pre = clf.predict(to_export['X_test_pre'])

    sample_train = shap.sample(to_export['X_train'], nsamples=10, random_state=90)

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
    end_time = (time.time() - start_time) / 60

    # Make shap plot
    plt.figure()
    shap.summary_plot(shap_values, to_export['X_test_post'], feature_names=all_cols, plot_type="bar",  show=False)
    plt.title('SHAP summary plot - REGRESSION')
    plt.tight_layout()
    plt.savefig('images/RF_regression_Summary_plot_%s' % filename)

    diz_rf = {
              "oob_score": clf.oob_score_,
              'Random Forest feature importance': sort_ord_zip,

              'Mean Absolute Error (MAE) pre drift:': metrics.mean_absolute_error(to_export['y_test_pre'],
                                                                        prediction_pre),
              'Mean Absolute Error (MAE) post drift:': metrics.mean_absolute_error(to_export['y_test_post'],
                                                                  prediction_post),

              'Mean Squared Error (MSE) pre drift:': metrics.mean_squared_error(to_export['y_test_pre'],
                                                                      prediction_pre),
              'Mean Squared Error (MSE) post drift:': metrics.mean_squared_error(to_export['y_test_post'],
                                                                prediction_post),

              'Root Mean Squared Error (RMSE) pre drift:': metrics.mean_squared_error(to_export['y_test_pre'],
                                                                            prediction_pre, squared=False),
              'Root Mean Squared Error (RMSE) post drift:': metrics.mean_squared_error(to_export['y_test_post'],
                                                                      prediction_post, squared=False),

              'Explained Variance Score pre drift:': metrics.explained_variance_score(to_export['y_test_pre'],
                                                                            prediction_pre),
              'Explained Variance Score post drift:': metrics.explained_variance_score(to_export['y_test_post'],
                                                                      prediction_post),

              'Max Error pre drift:': metrics.max_error(to_export['y_test_pre'], prediction_pre),
              'Max Error post drift:': metrics.max_error(to_export['y_test_post'], prediction_post),

              'R^2 pre drift:': metrics.r2_score(to_export['y_test_pre'], prediction_pre),
              'R^2 post drift:': metrics.r2_score(to_export['y_test_post'], prediction_post),

              'time': end_time

              }

    with open('other_files/RF_METRICS_REGRESSION_POST_%s.json' % filename, 'w', encoding='utf-8') as f1:
        json.dump(diz_rf, f1, cls=NumpyEncoder)
    f1.close()

    return clf
