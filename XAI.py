import warnings
warnings.filterwarnings("ignore")

import shap
import lime
import lime.lime_tabular
from alibi.explainers import AnchorTabular
import matplotlib.pyplot as plt
import numpy as np
from numpyencoder import NumpyEncoder
import pandas as pd
import json
import time


first_time = time.time() # returns the processor time


def d3_xai(data_for_xai, cols, all_cols, filename):
    '''
    Function for xai methods with D3 approach

    cols : swapped columns
    all_cols : all columns
    '''

    ret = []
    ret.append({'swapped columns': cols})
    ret.append({'columns': all_cols})

    lime_res = []
    lime_res.append({'swapped columns': cols})
    lime_res.append({'columns': all_cols})

    anchor_res = []
    anchor_res.append({'swapped columns': cols})
    anchor_res.append({'columns': all_cols})

    time_shap = []
    time_lime = []
    time_anchor = []

    diz_shap = []
    auc_values = []

    for diz in data_for_xai[-1:][:20]:
        diz_shap.append(diz)

        train_set = pd.DataFrame(diz['X_train'], columns=all_cols)
        test_set = pd.DataFrame(diz['X_test'], columns=all_cols)
        class_names = np.unique(diz['y_train'])
        auc_values.append(diz['AUC'])

        k = 0

        acc_tr, acc_test, prec_post, rec_post, f_post = diz['Accuracy_train'], diz['Accuracy_test'], diz['Precision_post'], diz['Recall_post'], diz['F1_score_post']
        name = ['train_accuracy', 'test_accuracy', 'Precision_post', 'Recall_post', 'F1_score_post']
        d3_accuracy_df = pd.DataFrame([acc_tr, acc_test, prec_post, rec_post, f_post], index = name)
        d3_accuracy_df.to_excel(f'other_files/D3_ACCURACY_{filename}.xlsx')

        # Setting explainers
        explainer_shap = shap.KernelExplainer(diz['model'].predict_proba,
                                              shap.sample(train_set, nsamples=100, random_state=90),
                                              random_state=90,
                                              link='identity',
                                              l1_reg=len(all_cols)
                                              )

        explainer_lime = lime.lime_tabular.LimeTabularExplainer(diz['X_train'],
                                                                mode='classification',
                                                                feature_names=all_cols,
                                                                feature_selection='none',
                                                                discretize_continuous=True,
                                                                discretizer='quartile',
                                                                verbose=True)
        predict_fn = lambda x:diz['model'].predict(x)
        explainer_anchor = AnchorTabular(predict_fn, all_cols)
        explainer_anchor.fit(diz['X_train'], disc_perc=(25, 50, 75))

        # Start explanations
        for i in range(len(diz['X_test'])):
            pred = diz['predictions'][i]
            predict_proba = diz['model'].predict_proba(diz['X_test'][i].reshape(1, -1))[0]

            # SHAP
            start_time_sh = time.time()
            shap_values = explainer_shap.shap_values(test_set.iloc[i, :])
            end_time_shap = (time.time() - start_time_sh) / 60
            time_shap.append(end_time_shap)

            zipped = list(zip(shap_values[pred], all_cols))
            ordered_shap_list = sorted(zipped, key=lambda x: abs(x[0]), reverse=True)
            model_output = (explainer_shap.expected_value[1] + shap_values[1].sum()).round(4)  # è uguale a ML prediction

            # Get the force plot for each row
            shap.initjs()
            plt.figure(figsize=(16, 5))
            shap.plots.force(explainer_shap.expected_value[1], shap_values[1],test_set.iloc[i,:], link = 'logit',feature_names=all_cols, matplotlib = True, show=False, text_rotation=6) # matplotlib = True,
            plt.savefig('html_images/'+ f'D3_SHAP_row_{str(i)}_dataset_{filename}.jpg', bbox_inches='tight')

            swap_shap = [(tup[1], True if tup[1] in cols else False) for tup in ordered_shap_list]
            feat_shap_val = [(tup[1], round(tup[0], 3)) for tup in ordered_shap_list]

            dizio = {'batch %s' % k: {'row %s' % i: {
                'auc': diz['AUC'],
                'class_names': class_names,
                'model_output': model_output,
                'd3_prediction': pred,                                          # D3 class predicted
                'd3_pred_probs': predict_proba,                                 # D3 probs predicted
                'value_ordered': feat_shap_val,                                 # (feature, shap_value) : ordered list
                'swapped': swap_shap,                                           # (feature, bool) : ordered list of swapped variables
                'shap_values': shap_values
            }}}

            ret.append(dizio)

            # LIME 
            start_time_lime = time.time()
            exp_lime = explainer_lime.explain_instance(diz['X_test'][i],
                                                       diz['model'].predict_proba,
                                                       num_samples= 100,
                                                       num_features=len(all_cols),
                                                       distance_metric='euclidean')
            tot_lime = time.time() - start_time_lime
            time_lime.append(tot_lime)

            big_lime_list = exp_lime.as_list()  # list of tuples (representation, weight),
            ord_lime = sorted(big_lime_list, key=lambda x: abs(x[1]), reverse=True)

            exp_lime.as_pyplot_figure().tight_layout()
            plt.savefig('images/' +f' D3_LIME_row{str(i)}_dataset_{filename}')
            exp_lime.save_to_file('html_images/' + f'D3_lime_row_{str(i)}_dataset_{filename}.html')

            variables = []  # (name, real value)
            f_weight = []  # (feature, weight)
            swap = []  # (feature, bool)

            for t in ord_lime:
                tt = t[0].split(' ')
                if len(tt) == 3:
                    f_weight.append((tt[0], round(t[1], 3)))

                    if tt[0] in cols:
                        swap.append((tt[0], True))
                    else:
                        swap.append((tt[0], False))
                    variables.append((tt[0], round(float(tt[-1]), 3)))

                elif len(tt) > 3:
                    f_weight.append((tt[2], round(t[1], 3)))

                    if tt[2] in cols:
                        swap.append((tt[2], True))
                    else:
                        swap.append((tt[2], False))

                    mean_sum = round((float(tt[0]) + float(tt[-1])) / 2, 3)  #
                    variables.append((tt[2], mean_sum))

            lime_diz = {'batch %s' % k: {'row %s' % i: {
                'auc': diz['AUC'],
                'class_names': class_names,
                'd3_prediction': pred,                                  # D3 prediction (LR)
                'd3_pred_probs': predict_proba,                         # D3 prediction (LR)
                'lime_pred': exp_lime.local_pred,
                'value_ordered': f_weight,                              # (feature, lime_weight)
                'feature_value': variables,                             # (feature, real value)
                'swapped': swap                                         # (feature, bool)
            }}}

            lime_res.append(lime_diz)


            # ANCHORS
            print('row id ', i)
            start_time_anchor = time.time()

            exp_anchor = explainer_anchor.explain(diz['X_test'][i],
                                                  threshold=0.90,
                                                  beam_size=len(all_cols),
                                                  coverage_samples=1000)
            end_time_a = time.time() - start_time_anchor
            time_anchor.append(end_time_a)
            print('\n ALIBI EXPLANATION \n', exp_anchor)

            rules = exp_anchor.anchor
            print('RULE', rules)
            precision = exp_anchor.precision
            coverage = exp_anchor.coverage

            contrib = []
            swapped = []

            if len(rules) == 0:
                if len(contrib) > 0:
                    contrib.append(contrib[-1])
                else:
                    contrib.append(
                        'empty rule: all neighbors have the same label')  # al primo batch potrebbe essere vuoto
            else:
                for s in rules:  # nel caso in cui ci siano più predicati
                    splittato = s.split(' ')  # splittato = [nswprice, >, 0.08], [0.3 <= feature <= 0.6]
                    n = len(splittato)

                    if n == 3:  # 1 feature: caso tipo [feature <= 0.5]
                        contrib.append(splittato[0])
                        if splittato[0] in cols:
                            swapped.append((splittato[0], True))
                        else:
                            swapped.append((splittato[0], False))
                    if n > 3:
                        for el in splittato:
                            if el.isalpha() and el in cols:
                                contrib.append(el)
                                swapped.append((el, True))

                            elif el.isalpha() and not el in cols:
                                contrib.append(el)
                                swapped.append((el, False))

                            else:  # caso tipo: 0.3 <= feature <= 0.6
                                pos = 2
                                contrib.append(splittato[pos])
                                if splittato[pos] in cols:
                                    swapped.append((splittato[pos], True))
                                    break
                                else:
                                    swapped.append((splittato[pos], False))
                                    break

            diz_anchors = {'batch %s' % k: {'row %s' % i: {
                'auc': diz['AUC'],
                'class_names': class_names,
                'd3_prediction': pred,
                'd3_pred_probs': predict_proba,
                'Anchor_prediction': class_names[explainer_anchor.predictor(diz['X_test'][i].reshape(1, -1))[0]],
                'rule': ' AND '.join(exp_anchor.anchor),
                'precision': precision,
                'coverage': coverage,
                'swapped': swapped,
                'value_ordered': contrib,
            }}}

            anchor_res.append(diz_anchors)

        k += 1

    mean_time_shap = np.mean(time_shap)
    mean_time_lime = np.mean(time_lime)
    mean_time_anchor = np.mean(time_anchor)
    index = ['mean_time_shap', 'mean_time_lime', 'mean_time_anchor']
    means = [mean_time_shap, mean_time_lime, mean_time_anchor]
    to_export = pd.DataFrame(means, columns=['mean time'], index=index)
    to_export.to_excel(f'other_files/D3_TIME_{filename}.xlsx')

    # D3 FILES
    with open('results/' + 'D3_SHAP_%s.json' % filename, 'w', encoding='utf-8') as f:
        json.dump(ret, f, cls=NumpyEncoder)

    with open('results/' + 'D3_LIME_%s.json' % filename, 'w', encoding='utf-8') as f1:
        json.dump(lime_res, f1, cls=NumpyEncoder)

    with open('other_files/' + 'D3_ANCHORS_%s.json' % filename, 'w', encoding='utf-8') as ff2:
        json.dump(anchor_res, ff2, cls=NumpyEncoder)

    f.close()
    f1.close()
    ff2.close()




def st_xai(data_for_xai, cols, all_cols, filename):
    '''
    Function for xai methods with student-teacher approach

    cols : swapped columns
    all_cols : all columns
    '''

    k = 0

    ret_st = []
    ret_st.append({'swapped columns': cols})
    ret_st.append({'columns': all_cols})

    lime_res_st = []
    lime_res_st.append({'swapped columns': cols})
    lime_res_st.append({'columns': all_cols})

    anchor_res_st = []
    anchor_res_st.append({'swapped columns': cols})
    anchor_res_st.append({'columns': all_cols})

    time_shap = []
    time_lime = []
    time_anchor = []

    student_error = []

    for diz in data_for_xai:
        print('---- ST------')
        print(diz['drifted'])
        train_set = pd.DataFrame(diz['X_train'], columns=all_cols)
        test_set = pd.DataFrame(diz['X_test'], columns=all_cols)
        student_error.append(diz['student_error'])

        class_names = np.unique(diz['y_train'])

        explainer_shap = shap.KernelExplainer(diz['model'].predict_proba,
                                              shap.sample(train_set, nsamples=100, random_state=90),
                                              link='identity',
                                              l1_reg = len(all_cols)
                                              )

        explainer_lime = lime.lime_tabular.LimeTabularExplainer(diz['X_train'],
                                                                mode='classification',
                                                                feature_names=all_cols,
                                                                feature_selection='none',
                                                                discretize_continuous=True,
                                                                discretizer='quartile',
                                                                verbose=True
                                                                )

        predict_fn = lambda x: diz['model'].predict(x)
        explainer_anchor = AnchorTabular(predict_fn, all_cols)
        explainer_anchor.fit(diz['X_train'], disc_perc=(25, 50, 75))

        pred = int(diz['class_student'])
        predict_proba = diz['model'].predict_proba(diz['X_test'][0].reshape(1, -1))[0]  # default: l2 penalty = ridge regression

        # SHAP
        start_time = time.time()
        print('ST Computing shap values')
        shap_values = explainer_shap.shap_values(test_set) # test set è una riga
        end_time_shap = (time.time() - start_time) / 60
        time_shap.append(end_time_shap)

        zipped = list(zip(shap_values[1][0], all_cols))
        ordered_shap_list = sorted(zipped, key=lambda x: abs(x[0]), reverse=True)
        model_output = (explainer_shap.expected_value[1] + shap_values[1].sum()).round(4)  # è uguale a ML prediction

        # Get the force plot for each row
        shap.initjs()
        shap.plots.force(explainer_shap.expected_value[1], shap_values[1], test_set, link='logit', feature_names=all_cols, show=False, matplotlib=True, text_rotation=6)
        plt.savefig('html_images/'+ f'ST_SHAP_row{str(k)}_dataset_{filename}', bbox_inches='tight')

        swap_shap = [(tup[1], True if tup[1] in cols else False) for tup in ordered_shap_list]
        feat_shap_val = [(tup[1], tup[0]) for tup in ordered_shap_list]

        dizio = {'batch %s' % k: {'row %s' % k: {
            'student_error': diz['student_error'],
            'class_names': class_names,
            'drift': diz['drifted'],
            'model_output': model_output,
            'ST_prediction': pred,
            'ST_pred_probs': predict_proba,             # ST prediction
            'value_ordered': feat_shap_val,             # (feature, shap_value)
            'swapped': swap_shap,                       # (feature, bool),
            'shap_values': shap_values
        }}}
        ret_st.append(dizio)

        # LIME
        start_time_lime = time.time()
        exp_lime = explainer_lime.explain_instance(diz['X_test'][0],
                                                   diz['model'].predict_proba,
                                                   num_samples=100,
                                                   distance_metric='euclidean') # provare anche una distanza diversa
        tot_lime = time.time() - start_time_lime
        time_lime.append(tot_lime)

        big_lime_list = exp_lime.as_list()  # list of tuples (representation, weight),
        ord_lime = sorted(big_lime_list, key=lambda x: abs(x[1]), reverse=True)
        exp_lime.as_pyplot_figure().tight_layout()
        plt.savefig('images/' + f' ST_LIME_row{str(k)}_dataset_{filename}')
        exp_lime.save_to_file('html_images/' + f'ST_lime_row_{str(k)}_dataset_{filename}.html')

        variables = []  # (name, real value)
        f_weight = []  # (feature, weight)
        swap = []  # (feature, bool)

        for t in ord_lime:
            tt = t[0].split(' ')
            if len(tt) == 3:
                f_weight.append((tt[0], round(t[1], 3)))

                if tt[0] in cols:
                    swap.append((tt[0], True))
                else:
                    swap.append((tt[0], False))
                variables.append((tt[0], round(float(tt[-1]), 3)))

            elif len(tt) > 3:
                f_weight.append((tt[2], round(t[1], 3)))

                if tt[2] in cols:
                    swap.append((tt[2], True))
                else:
                    swap.append((tt[2], False))

                mean_sum = round((float(tt[0]) + float(tt[-1])) / 2, 3)
                variables.append((tt[2], mean_sum))

        lime_diz = {'batch %s' % k: {'row %s' % k: {
            'student_error': diz['student_error'],
            'class_names': class_names,
            'drift': diz['drifted'],
            'lime_pred':  exp_lime.local_pred,
            'ST_prediction': pred,                              # ST prediction
            'ST_pred_probs': predict_proba,                     # ST predicted probs
            'value_ordered': f_weight,                          # (feature, lime_weight)
            'feature_value': variables,                         # (feature, real value)
            'swapped': swap                                     # (feature, bool)

        }}}

        lime_res_st.append(lime_diz)

        # ANCHORS
        print('row id ', k)
        start_time_anch = time.time()
        exp_anchor = explainer_anchor.explain(diz['X_test'][0],
                                              threshold=0.90,
                                              beam_size=len(all_cols),
                                              coverage_samples=1000)
        end_time_a = time.time() - start_time_anch
        time_anchor.append(end_time_a)

        rules = exp_anchor.anchor
        precision = exp_anchor.precision
        coverage = exp_anchor.coverage

        contrib = []
        swapped = []

        if len(rules) == 0:
            if len(contrib) > 0:
                contrib.append(contrib[-1])
            else:
                contrib.append('empty rule: all neighbors have the same label')  # al primo batch potrebbe essere vuoto
        else:
            for s in rules:
                splittato = s.split(' ')
                n = len(splittato)

                if n == 3:  # 1 feature: caso tipo [feature <= 0.5]
                    contrib.append(splittato[0])
                    if splittato[0] in cols:
                        swapped.append((splittato[0], True))
                    else:
                        swapped.append((splittato[0], False))
                if n > 3:
                    for el in splittato:
                        if el.isalpha() and el in cols:
                            contrib.append(el)
                            swapped.append((el, True))

                        elif el.isalpha() and not el in cols:
                            contrib.append(el)
                            swapped.append((el, False))

                        else:
                            pos = 2
                            contrib.append(splittato[pos])
                            if splittato[pos] in cols:
                                swapped.append((splittato[pos], True))
                                break
                            else:
                                swapped.append((splittato[pos], False))
                                break

        diz_anchors = {'batch %s' % k: {'row %s' % k: {
            'student_error': diz['student_error'],
            'class_names': class_names,
            'ST_prediction': pred,                       # ST prediction
            'ST_pred_probs': predict_proba,              # ST predicted proba
            'rule': ' AND '.join(exp_anchor.anchor),
            'precision': precision,
            'coverage': coverage,
            'swapped': swapped,                          # (feature, bool)
            'value_ordered': contrib,               # list of features
            'Anchor_prediction': class_names[explainer_anchor.predictor(diz['X_test'][0].reshape(1, -1))[0]]
        }}}

        anchor_res_st.append(diz_anchors)

        k += 1

    print('number of identified drifted rows', k)

    mean_time_shap = np.mean(time_shap)
    mean_time_lime = np.mean(time_lime)
    mean_time_anchor = np.mean(time_anchor)
    index = ['mean_time_shap', 'mean_time_lime', 'mean_time_anchor']
    means = [mean_time_shap,mean_time_lime,mean_time_anchor]
    to_export = pd.DataFrame(means, columns=['mean time'], index=index)
    to_export.to_excel(f'other_files/ST_TIME_{filename}.xlsx')

    # ST FILES
    with open('results/' + 'ST_SHAP_%s.json' % filename, 'w', encoding='utf-8') as f:
        json.dump(ret_st, f, cls=NumpyEncoder)

    with open('results/' + 'ST_LIME_%s.json' % filename, 'w', encoding='utf-8') as f1:
        json.dump(lime_res_st, f1, cls=NumpyEncoder)

    with open('other_files/' + 'ST_ANCHORS_%s.json' % filename, 'w', encoding='utf-8') as f2:
        json.dump(anchor_res_st, f2, cls=NumpyEncoder)

    f.close()
    f1.close()
    f2.close()


tot_time = f"XAI.PY Total time: {(time.time() - first_time) / 60} minutes"
print(f"XAI.PY Total time: {(time.time() - first_time) / 60} minutes")
print('END XAI')


