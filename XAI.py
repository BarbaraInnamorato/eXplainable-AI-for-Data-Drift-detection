import warnings
warnings.filterwarnings("ignore")

import shap
import lime
import lime.lime_tabular
#from anchor import anchor_tabular

#import alibi
from alibi.explainers import AnchorTabular

import matplotlib.pyplot as plt
import numpy as np
from numpyencoder import NumpyEncoder

import pandas as pd
import json
import time
import os

# Connect pandas with plotly
import cufflinks as cf
cf.go_offline()
from plotly.offline import init_notebook_mode
init_notebook_mode(connected='true')



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

    for diz in data_for_xai[-1:]:
        train_set = pd.DataFrame(diz['X_train'], columns=all_cols)
        test_set = pd.DataFrame(diz['X_test'], columns=all_cols)

        class_names = np.unique(diz['y_train'])

        k = 0

        d3_train_acc = diz['model'].score(train_set, diz['y_train'])
        d3_test_acc = diz['model'].score(test_set, diz['y_test'])
        #print("D3 train accuracy: %0.3f" % diz['model'].score(train_set, diz['y_train']))
        #print("D3 test accuracy: %0.3f" % diz['model'].score(test_set, diz['y_test']))


        d3_accuracy_df = pd.DataFrame([d3_train_acc, d3_test_acc], index=['accuracy'])
        d3_accuracy_df.to_excel(f'other_files/D3_acc_{filename}.xlsx')

        # Setting explainers
        explainer_shap = shap.KernelExplainer(diz['model'].predict_proba,
                                              train_set,
                                              nsamples=100,
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
        for i in range(len(diz['X_test'])):  # da 0 a 66-1
            pred = diz['predictions'][i]
            predict_proba = diz['model'].predict_proba(diz['X_test'][i].reshape(1, -1))[0]


            #############################  SHAP  ##############################
            '''
            - le variabili con shap value negativo dovrebbero essere quelle che spingono verso zero 
            - feat_shap_val: le variabili sono ordinate in base al valore assoluto del rispettivo shap value
            - più lo shap_value in valore è assoluto è alto, più la variabile è importante 
            # mean = sum(abs(tup[0]) for tup in ordered_shap_list)/len(ordered_shap_list)
            '''
            start_time_sh = time.time()
            shap_values = explainer_shap.shap_values(test_set.iloc[i, :])
            end_time_shap = (time.time() - start_time_sh) / 60
            #time_shap1 = f"D3 - SHAP - Total time {filename}: {end_time_shap} minutes"

            # print(f"D3 - SHAP - Total time {filename}: {(time.time() - start_time) / 60} minutes")
            #time_shap.append({'start':start_time_sh, 'time':end_time_shap,'time_string': time_shap1, 'iter':i})
            time_shap.append(end_time_shap)
            model_output = (explainer_shap.expected_value + shap_values[pred].sum()).round(4) #list of probs
            class_pred = np.argmax(abs(model_output))
            print(f'SHAP model output {filename}', model_output)
            print(f'SHAP VALUES {filename}', shap_values)

            # questo mo_output è l'output della black-box
            #mo_output = (explainer_shap.expected_value[class_pred] + shap_values[class_pred].sum()).round(4)
            #print('mo output', mo_output)

            zipped = list(zip(shap_values[pred], all_cols))
            ordered_shap_list = sorted(zipped, key=lambda x: x[0], reverse=True)

            # Get the force plot for each row
            shap.initjs()
            fig = plt.figure()
            fig.set_figheight(6)
            shap.plots.force(explainer_shap.expected_value[class_pred], shap_values[class_pred],test_set.iloc[i,:], link = 'logit',feature_names=all_cols, show=False, matplotlib = True, text_rotation=6)#, figsize=(50,12))
            #plt.title(f'Local forceplot row {str(i)} dataset {filename}', position=(0.3, 0.7))
            plt.tight_layout()
            plt.savefig('images/'+ f'D3 Local SHAP row {str(i)} dataset {filename}')


            swap_shap = [(tup[1], True if tup[1] in cols else False) for tup in ordered_shap_list]
            feat_shap_val = [(tup[1], round(tup[0], 3)) for tup in ordered_shap_list]

            dizio = {'batch %s' % k: {'row %s' % i: {
                'class_names': class_names,
                'd3_prediction': pred,                                          # D3 class predicted
                'd3_pred_probs': predict_proba,                                 # D3 probs predicted
                'SHAP_probs': model_output,
                'is ML correct': class_pred == pred,
                'value_ordered': feat_shap_val,                                 # (feature, shap_value) : ordered list
                'swapped': swap_shap,                                           # (feature, bool) : ordered list of swapped variables
                'shap_prediction': class_pred,                                  # xai prediction : class
                'shap_values': shap_values
            }}}
            ret.append(dizio)


            #############################  LIME  ###########################
            '''
            - le variabili con |weight| sono quelle più determinati per la prediction del ML model
            - feature_weight_sorted: le variabili sono ordinate in base al valore assoluto del rispettivo peso calcolato con lime
            - feature_value: valore reale del feature nella riga considerata
            # mean = round(sum(abs(tup[1]) for tup in big_lime_list)/len(big_lime_list),3)
            '''
            start_time_lime = time.time()
            exp_lime = explainer_lime.explain_instance(diz['X_test'][i],
                                                       diz['model'].predict_proba,
                                                       num_samples= 100,
                                                       num_features=len(all_cols),
                                                       distance_metric='euclidean')  # provare anche una distanza diversa

            ###
            tot_lime = time.time() - start_time_lime
            time_lime.append(tot_lime)
            #time_lime1 = f"D3 - LIME - Total time: {(time.time() - start_time_lime) / 60} minutes"
            #print(f"D3 - LIME - Total time: {(time.time() - start_time) / 60} minutes")

            big_lime_list = exp_lime.as_list()  # list of tuples (representation, weight),
            ord_lime = sorted(big_lime_list, key=lambda x: abs(x[1]), reverse=True)

            lime_prediction = exp_lime.local_pred
            lime_class_pred = [0 if lime_prediction < 0.5 else 1][0]
            exp_lime.as_pyplot_figure().tight_layout()
            plt.savefig('images/' +f' D3 Local LIME row {str(i)} dataset {filename}')
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

                    mean_sum = round((float(tt[0]) + float(tt[-1])) / 2, 3)  # caso in cui il valore di un feature è in un range di valori
                    variables.append((tt[2], mean_sum))

            lime_diz = {'batch %s' % k: {'row %s' % i: {
                'class_names': class_names,
                'd3_prediction': pred,                                  # D3 prediction (LR)
                'd3_pred_probs': predict_proba,                         # D3 prediction (LR)
                'LIME_prediction': lime_class_pred,                     # xai prediction
                'LIME_LOCAL_prediction': lime_prediction,           # xai prediction
               # 'LIME_pred_probs': lime_probs,
                'value_ordered': f_weight,                              # (feature, lime_weight)
                'feature_value': variables,                             # (feature, real value)
                'swapped': swap                                         # (feature, bool)
            }}}

            lime_res.append(lime_diz)


            #############################  ANCHORS  ###########################
            start_time_anchor = time.time()
            class_list_str = ['0', '1']
            pred_uno = class_list_str[explainer_anchor.predictor(diz['X_test'][i].reshape(1, -1))[0]]
            print('ALIBI PREDICTION', pred_uno)

            exp_anchor = explainer_anchor.explain(diz['X_test'][i],
                                                  threshold=0.90,
                                                  beam_size=len(all_cols),
                                                  coverage_samples=1000)
            end_time_a = time.time() - start_time_anchor
            time_anchor.append(end_time_a)
            print('ALIBI EXPLANATION', exp_anchor)

            rules = exp_anchor.anchor
            print('RULES', rules)
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
                    if n > 3:  # more than 1 feature: caso tipo rules = ['nswprice > 0.08', 'vicprice > 0.00', 'day <= 2.00']
                        # pos = 0                                      # splittato = [nswprice, >, 0.08]
                        for el in splittato:
                            if el.isalpha() and el in cols:
                                contrib.append(el)
                                swapped.append((el, True))

                            elif el.isalpha() and not el in cols:
                                contrib.append(el)
                                swapped.append((el, False))

                            else:  # caso tipo: 0.3 <= feature <= 0.6
                                pos = 2
                                # print('splittato',splittato)
                                contrib.append(splittato[pos])
                                if splittato[pos] in cols:
                                    swapped.append((splittato[pos], True))
                                    break
                                else:
                                    swapped.append((splittato[pos], False))
                                    break

            diz_anchors = {'batch %s' % k: {'row %s' % i: {
                'class_names': class_names,
                'd3_prediction': pred,
                'd3_pred_probs': predict_proba,
                'Anchor_prediction': pred_uno,
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
    to_export.to_excel(f'other_files/D3 {filename}.xlsx')

    # D3 FILES
    with open('results/' + 'D3_SHAP_%s.json' % filename, 'w', encoding='utf-8') as f:
        json.dump(ret, f, cls=NumpyEncoder)

    with open('results/' + 'D3_LIME_%s.json' % filename, 'w', encoding='utf-8') as f1:
        json.dump(lime_res, f1, cls=NumpyEncoder)

    with open('results/' + 'D3_ANCHORS_%s.json' % filename, 'w', encoding='utf-8') as ff2:
        json.dump(anchor_res, ff2, cls=NumpyEncoder)

    #return ret, anchor_res, lime_res

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

    for diz in data_for_xai:
        #print('---- ST ------')
        train_set = pd.DataFrame(diz['X_train'], columns=all_cols)
        test_set = pd.DataFrame(diz['X_test'], columns=all_cols)

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

        #############################  SHAP  ##############################
        '''
        - le variabili con shap value negativo dovrebbero essere quelle che spingono verso zero 
        - feat_shap_val: le variabili sono ordinate in base al valore assoluto del rispettivo shap value
        - più lo shap_value in valore è assoluto è alto, più la variabile è importante 
        - #mean = sum(abs(tup[0]) for tup in ordered_shap_list)/len(ordered_shap_list)

        '''
        start_time = time.time()
        print('ST SHAP sto calcolando shap values')
        shap_values = explainer_shap.shap_values(test_set) # test set è una riga
        end_time_shap = (time.time() - start_time) / 60
        time_shap.append(end_time_shap)

        #expected_values = list(explainer_shap.expected_value)
        print(f'ST shap values {filename}', shap_values)
        model_output = (explainer_shap.expected_value + shap_values[pred].sum()).round(4)
        class_pred = np.argmax(abs(model_output))

        zipped = list(zip(shap_values[class_pred][0], all_cols))
        ordered_shap_list = sorted(zipped, key=lambda x: x[0], reverse=True)


        # Get the force plot for each row
        shap.initjs()
        shap.plots.force(explainer_shap.expected_value[class_pred], shap_values[class_pred], test_set, link='logit', feature_names=all_cols, show=False, matplotlib = True, text_rotation=6)#, figsize=(50,12))
        name = f'ST Local SHAP row {str(k)}, dataset {filename}'
        plt.title(f'Local SHAP row {str(k)} dataset {filename}')
        plt.tight_layout()
        plt.savefig('images/'+ name)

        swap_shap = [(tup[1], True if tup[1] in cols else False) for tup in ordered_shap_list]
        feat_shap_val = [(tup[1], tup[0]) for tup in ordered_shap_list]
        #model_output = (explainer_shap.expected_value[pred] + shap_values[0].sum()).round(4)

        dizio = {'batch %s' % k: {'row %s' % k: {
            'class_names': class_names,
            'ST_prediction': pred,
            'ST_pred_probs': predict_proba,             # ST prediction
            'is ML correct': class_pred == pred,
            'SHAP_prediction': class_pred,
            'value_ordered': feat_shap_val,             # (feature, shap_value)
            'swapped': swap_shap,                       # (feature, bool),
            'SHAP_probs': model_output,                 # xai prediction
            'shap_values' : shap_values
        }}}
        ret_st.append(dizio)

        ############################### LIME ################################
        start_time_lime = time.time()
        exp_lime = explainer_lime.explain_instance(diz['X_test'][0],
                                                   diz['model'].predict_proba,
                                                   num_samples=100,
                                                   distance_metric='euclidean') # provare anche una distanza diversa
        tot_lime = time.time() - start_time_lime
        time_lime.append(tot_lime)

        #time_lime2 = f"Total time: {(time.time() - start_time) / 60} minutes"
        #lime_probs = list(exp_lime.predict_proba)  # prob of being in class 0 or in class 1

        big_lime_list = exp_lime.as_list()  # list of tuples (representation, weight),
        ord_lime = sorted(big_lime_list, key=lambda x: abs(x[1]), reverse=True)
        lime_prediction = exp_lime.local_pred
        lime_class_pred = [0 if lime_prediction < 0.5 else 1][0]
        exp_lime.as_pyplot_figure().tight_layout()
        plt.text(0.3, 0.7, f' D3 Local LIME row {str(k)}')
        plt.savefig('images/' + f' ST Local LIME row {str(k)} dataset {filename}')
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

                mean_sum = round((float(tt[0]) + float(tt[-1])) / 2,
                                 3)  # caso in cui il valore di un feature è in un range di valori
                variables.append((tt[2], mean_sum))

        lime_diz = {'batch %s' % k: {'row %s' % k: {
            'class_names': class_names,
            'ST_prediction': pred,                              # ST prediction
            'ST_pred_probs': predict_proba,                     # ST predicted probs
            'LIME_LOCAL_prediction': lime_prediction,       # xai prediction
            'LIME_prediction': lime_class_pred,                 # xai prediction
            'value_ordered': f_weight,                          # (feature, lime_weight)
            'feature_value': variables,                         # (feature, real value)
            'swapped': swap                                     # (feature, bool)

        }}}

        lime_res_st.append(lime_diz)

        ################### ANCHORS #########################################
        start_time_anch = time.time()
        class_list_str = ['0', '1']
        pred_uno = class_list_str[explainer_anchor.predictor(diz['X_test'][k].reshape(1, -1))[0]]
        print('ALIBI PREDICTION', pred_uno)

        exp_anchor = explainer_anchor.explain(diz['X_test'][k],
                                              threshold=0.90,
                                              beam_size=len(all_cols),
                                              coverage_samples=1000)
        end_time_a = time.time() - start_time_anch
        time_anchor.append(end_time_a)
        print('ALIBI EXPLANATION', exp_anchor)

        rules = exp_anchor.anchor
        print('RULES', rules)
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
            for s in rules:  # nel caso in cui ci siano più predicati
                splittato = s.split(' ')  # splittato = [nswprice, >, 0.08], [0.3 <= feature <= 0.6]
                n = len(splittato)

                if n == 3:  # 1 feature: caso tipo [feature <= 0.5]
                    contrib.append(splittato[0])
                    if splittato[0] in cols:
                        swapped.append((splittato[0], True))
                    else:
                        swapped.append((splittato[0], False))
                if n > 3:  # more than 1 feature: caso tipo rules = ['nswprice > 0.08', 'vicprice > 0.00', 'day <= 2.00']
                                                         # splittato = [nswprice, >, 0.08]
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

        diz_anchors = {'batch %s' % k: {'row %s' % k: {
            'class_names': class_names,
            'ML_prediction': pred,                       # ST prediction
            'ML_pred_probs': predict_proba,              # ST predicted proba
            'Anchor_prediction': pred_uno,
            'rule': ' AND '.join(exp_anchor.anchor),
            'precision': precision,
            'coverage': coverage,
            'swapped': swapped,                          # (feature, bool)
            'value_ordered': contrib,               # list of features
        }}}

        anchor_res_st.append(diz_anchors)

        k += 1

    mean_time_shap = np.mean(time_shap)
    mean_time_lime = np.mean(time_lime)
    mean_time_anchor = np.mean(time_anchor)
    index = ['mean_time_shap', 'mean_time_lime', 'mean_time_anchor']
    means = [mean_time_shap,mean_time_lime,mean_time_anchor]
    to_export = pd.DataFrame(means, columns=['mean time'], index=index)
    to_export.to_excel(f'other_files/ST_{filename}.xlsx')

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

    #return ret, lime_res, anchor_res


tot_time = f"XAI.PY Total time: {(time.time() - first_time) / 60} minutes"
print(f"XAI.PY Total time: {(time.time() - first_time) / 60} minutes")
print('END XAI')


"""
from pycebox.ice import ice, ice_plot
def ice_plot(data_for_xai, cols, all_cols):
"""