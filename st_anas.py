import shap
import lime
import lime.lime_tabular
#from anchor import anchor_tabular
import matplotlib.pyplot as plt
import numpy as np
from numpyencoder import NumpyEncoder
from alibi.explainers import AnchorTabular
from anchor import anchor_tabular

import pandas as pd
import json
import time

# from sklearn.ensemble import RandomForestRegressor
first_time = time.time() # returns the processor time

"""
def d3_xai(data_for_xai, cols, all_cols, filename):
    '''
    Function for xai methods with D3 approach

    cols : swapped columns
    all_cols : all columns
    '''

    ret = []
    ret.append({'swapped columns': cols})
    ret.append({'columns':  all_cols})

    lime_res = []
    lime_res.append({'swapped columns': cols})
    lime_res.append({'columns': all_cols})

    anchor_res = []
    anchor_res.append({'swapped columns': cols})
    anchor_res.append({'columns': all_cols})

    for diz in data_for_xai[-1:]:
        #len_test = len(diz['X_test'])

        train_set = pd.DataFrame(diz['X_train'], columns=all_cols)
        test_set = pd.DataFrame(diz['X_test'], columns=all_cols)
        class_names = np.unique(diz['y_train'])
        k = 0

        print("D3 train accuracy: %0.3f" % diz['model'].score(train_set, diz['y_train']))
        print("D3 test accuracy: %0.3f" % diz['model'].score(test_set, diz['y_test']))
        print('D3 MSError when predicting the mean \n', np.mean((diz['y_test'].mean() - diz['y_test']) ** 2))

        #k += len_test

        # print('D3 MSError \n', np.mean((diz['model'].predict(diz['X_test']) - diz['y_test']) ** 2,'\n\n'))


        #predict_for_anchors = lambda x: diz['model'].predict_proba(x)  # COMMON PREDICTION FUNCTION (TO CHECK)

        explainer_shap = shap.KernelExplainer(diz['model'].predict, shap.sample(train_set,nsamples=100, random_state=0))

        explainer_lime = lime.lime_tabular.LimeTabularExplainer(diz['X_train'],
                                                                mode='regression',
                                                                feature_names=all_cols,
                                                                feature_selection='auto',
                                                                discretize_continuous=True,
                                                                verbose=False)

        explainer_anchor = anchor_tabular.AnchorTabularExplainer(#predict_for_anchors,
                                                                 #diz['model'].predict,
                                                                 class_names=diz['class_names'],
                                                                 feature_names=all_cols,
                                                                 train_data=diz['X_train'],
                                                                 discretizer='quartile')


        for i in range(len(diz['X_test'])):  # da 0 a 66-1
            pred = diz['predictions'][i] # dal dizio creato in D3
            #predict_proba = diz['model'].predict(diz['X_test'][i].reshape(1, -1))[0]
            #print('D3 ANAS: is pred == model predict ?', pred == predict_proba)


            #  SHAP D3
            '''
            - le variabili con shap value negativo dovrebbero essere quelle che spingono verso zero 
            - feat_shap_val: le variabili sono ordinate in base al valore assoluto del rispettivo shap value
            - più lo shap_value in valore è assoluto è alto, più la variabile è importante 
            '''
            start_time = time.time()
            shap_values = explainer_shap.shap_values(diz['X_test'][i])
            print('%s -D3  - SHAP' % filename)
            print(f"Total time: {(time.time() - start_time) / 60} minutes")
            zipped = list(zip(shap_values, all_cols))
            ordered_shap_list = sorted(zipped, key=lambda x: x[0], reverse=True)

            # Get the force plot for each row
            shap.initjs()
            shap.plots.force(explainer_shap.expected_value, shap_values,diz['X_test'][i],feature_names=all_cols, show=False, matplotlib = True, text_rotation=6)
            plt.title('%s Local Force plot row %s '%(filename,i))
            plt.tight_layout()
            plt.savefig('images/'+ f'Local forceplot row {str(i)} dataset {filename}')

            # plt.close()

            swap_shap = [(tup[1], True if tup[1] in cols else False) for tup in ordered_shap_list]
            feat_shap_val = [(tup[1], round(tup[0], 3)) for tup in ordered_shap_list]
            higher_lower = [(tup[1], 'up' if tup[0] > 0.0 else 'down') for tup in ordered_shap_list]
            model_output = (explainer_shap.expected_value + shap_values[0].sum()).round(4)

            dizio = {'batch %s' % k: {'row %s' % i: {

                'ML_prediction': pred,                  # D3 prediction (LR)
                'value_ordered': feat_shap_val,         # (feature, shap_value)
                'swapped': swap_shap,                   # (feature, bool),
                'shap_prediction': model_output,        # xai prediction
                'high_low': higher_lower,                # pushing prediction to 1(higher) or to 0 (lower)
                'shap_values': shap_values
            }}}
            ret.append(dizio)



            #  LIME
            '''
            - le variabili con |weight| sono quelle più determinati per la prediction del ML model
            - feature_weight_sorted: le variabili sono ordinate in base al valore assoluto del rispettivo peso calcolato con lime
            - feature_value: valore reale del feature nella riga considerata
  
            '''
            start_time = time.time()
            exp_lime = explainer_lime.explain_instance(diz['X_test'][i],
                                                       diz['model'].predict,
                                                       num_features=len(all_cols),
                                                       distance_metric='euclidean')  # provare anche una distanza diversa
            print('%s - D3 - LIME' % filename)
            print(f"Total time: {(time.time() - start_time) / 60} minutes")
            # exp_lime.show_in_notebook(show_table = True)
            big_lime_list = exp_lime.as_list()  # list of tuples (representation, weight),
            ord_lime = sorted(big_lime_list, key=lambda x: abs(x[1]), reverse=True)

            variables = []  # (name, real value)
            f_weight = []   # (feature, weight)
            swap = []       # (feature, bool)

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

            lime_diz = {'batch %s' % k: {'row %s' % i: {
                'ML_prediction': pred,
                'LIME_LOCAL_prediction': exp_lime.local_pred,           # xai prediction
              # 'LIME_GLOBAL_prediction': exp_lime.predicted_value,
                'value_ordered': f_weight,                              # (feature, lime_weight)
                'feature_value': variables,                             # (feature, real value)
                'swapped': swap                                         # (feature, bool)
            }}}

            lime_res.append(lime_diz)



            '''
            #############################  ANCHORS  ###########################
            
             An anchor is a sufficient condition - that is, when the anchor holds, the prediction should be the same as the prediction for this instance.
  
              explainer.explain_instance:
            - threshold: the previously discussed minimal confidence level. threshold defines the minimum fraction of samples for a candidate anchor that need to 
                  lead to the same prediction as the original instance. A higher value gives more confidence in the anchor, but also leads to more computation time. 
                  The default value is 0.95.
            - tau: determines when we assume convergence for the multi-armed bandit. A bigger value for tau means faster convergence but also looser anchor conditions.
                  By default equal to 0.15.
            - beam_size: the size of the beam width. A bigger beam width can lead to a better overall anchor at the expense of more computation time.
            - batch_size: the batch size used for sampling. A bigger batch size gives more confidence in the anchor, again at the expense of computation time since
                  it involves more model prediction calls. The default value is 100.
            - coverage_samples: number of samples used to compute the coverage of the anchor. By default set to 10000.
  
            We set the precision threshold to 0.95. This means that predictions on observations where the anchor holds will be the same as the prediction on 
            the explained instance at least 95% of the time.
  
            https://github.com/marcotcr/anchor/blob/master/notebooks/Anchor%20on%20tabular%20data.ipynb
            '''
            print('diz',diz)
            start_time = time.time()
            exp_anchor = explainer_anchor.explain_instance(diz['X_test'][i],
                                                           diz['model'].predict,
                                                           threshold=0.90,
                                                           beam_size=len(all_cols))



            print('%s - D3 - ANCHOR' % filename)
            print(f"Total time: {(time.time() - start_time) / 60} minutes")
            #anchor_pred = [0,1][explainer_anchor.predictor(diz['X_test'][i].reshape(1, -1))[0]]
            #anchor_pred = exp_anchor.class_names[diz['model'].predict(diz['X_test'][i].reshape(1, -1))[0]]
            #anchor_pred = class_names[diz['model'].predict(diz['X_test'][i].reshape(1, -1))[0]]
            anchor_pred = explainer_anchor.class_names[diz['model'].predict(diz['X_test'][i].reshape(1, -1))[0]]

            #prediction = diz['model'].predict(diz['X_test'][i].reshape(1, -1))[0]
            print('ANCHOR PREDICTION', anchor_pred)
            #exp_anchor.show_in_notebook()
            #exp_anchor.examples(only_different_prediction = True) #np.ndarray

            rules = exp_anchor.names()
            precision = round(exp_anchor.precision(), 3)
            coverage = round(exp_anchor.coverage(), 3)
            '''
            print('anchor: %s' % (' AND '.join(exp_anchor.names())))
            print('precision: %.2f' % exp_anchor.precision())
            print('coverage: %.2f' % exp_anchor.coverage())
            
            # Get test examples where the anchora applies
            fit_anchor = np.where(np.all(diz['X_test'][i:, exp_anchor.features()] == diz['X_test'][i][exp_anchor.features()], axis=1))[0]
            print('Anchor test precision: %.2f' % (np.mean(diz['model'].predict(diz['X_test'][fit_anchor]) == diz['model'].predict(diz['X_test'][i].reshape(1, -1)))))
            print('Anchor test coverage: %.2f' % (fit_anchor.shape[0] / float(test_set.shape[0])))    
            print()'''

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

                'ML_prediction': pred,                      # ML prediction (class for D3)
                'Anchor_prediction': anchor_pred,            #
                'rule': ' AND '.join(exp_anchor.names()),
                'precision': precision,
                'coverage': coverage,
                'swapped': swapped,
                'value_ordered': contrib,
            }}}

            anchor_res.append(diz_anchors)



        k += 1






    # D3 FILES
    with open('results/' + 'D3_SHAP_REGRESSION_%s.json' % filename, 'w', encoding='utf-8') as f:
        json.dump(ret, f, cls=NumpyEncoder)

    with open('results/' + 'D3_LIME_REGRESSION_%s.json' % filename, 'w', encoding='utf-8') as f1:
        json.dump(lime_res, f1, cls=NumpyEncoder)

    #with open('other_files/' + 'D3_ANCHORS_%s.json' % filename, 'w', encoding='utf-8') as ff2:
    with open('results/' + 'D3_ANCHORS_%s.json' % filename, 'w', encoding='utf-8') as ff2:
        json.dump(anchor_res, ff2, cls=NumpyEncoder)

    #return ret, anchor_res, lime_res
"""



def st_xai(data_for_xai, cols, all_cols, filename):
        """
        Function for xai methods with student-teacher approach

        cols : swapped columns
        all_cols : all columns
        """

        k = 0
        ret_st = []
        ret_st.append('swapped columns %s' % cols)
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
            #train_sample = shap.sample(train_set, nsamples=100, random_state=0),  # se no train_set

            # Setting explainers

            explainer_shap = shap.KernelExplainer(diz['model'].predict,
                                                  shap.sample(train_set, nsamples=100, random_state=90),
                                                  link = 'identity',
                                                  l1_reg = len(all_cols)
                                                  )

            explainer_lime = lime.lime_tabular.LimeTabularExplainer(diz['X_train'],
                                                                    mode='regression',
                                                                    feature_names=all_cols,
                                                                    feature_selection='none',
                                                                    discretize_continuous=True,
                                                                    verbose=True)

            # explainer_anchor = anchor_tabular.AnchorTabularExplainer(#predict_for_anchors,
            #                                                          class_names=class_names,
            #                                                          #diz['model'].predict,
            #                                                          feature_names=all_cols,
            #                                                          train_data=diz['X_train'],
            #                                                          discretizer='quartile')
            predict_fn = lambda x: diz['model'].predict(x)
            explainer_anchor = AnchorTabular(predict_fn, all_cols)
            explainer_anchor.fit(diz['X_train'], disc_perc=(25, 50, 75))

            pred = diz['probs_student']

            #############################  SHAP  ##############################
            '''
            - le variabili con shap value negativo dovrebbero essere quelle che spingono verso zero 
            - feat_shap_val: le variabili sono ordinate in base al valore assoluto del rispettivo shap value
            - più lo shap_value in valore è assoluto è alto, più la variabile è importante 
            - #mean = sum(abs(tup[0]) for tup in ordered_shap_list)/len(ordered_shap_list)

            '''

            start_time_s = time.time()
            print('ST SHAP sto calcolando shap values')
            shap_values = explainer_shap.shap_values(test_set,  nsamples=100)  # test set è una riga
            print('ST ANAS SHAP VALUES', shap_values)
            end_time_shap = (time.time() - start_time_s) / 60
            time_shap.append(end_time_shap)

            #print(f"- ST - SHAP Total time {filename}: {(time.time() - start_time_s) / 60} minutes")
            #st_time_shap1 = f"- ST - SHAP Total time {filename}: {(time.time() - start_time_s) / 60} minutes"

            zipped = list(zip(shap_values[0], all_cols))
            ordered_shap_list = sorted(zipped, key=lambda x: x[0], reverse=True)

            # Get the force plot for each row
            shap.initjs()
            shap.plots.force(explainer_shap.expected_value, shap_values, test_set , feature_names=all_cols, show=False, matplotlib = True, text_rotation=6)#, figsize=(50,12))
            #plt.title(f'Local Force plot row {str(k)}')
            plt.tight_layout()
            plt.savefig('images/'+ f'ST Local forceplot row {str(k)} dataset {filename}')

            swap_shap = [(tup[1], True if tup[1] in cols else False) for tup in ordered_shap_list]
            feat_shap_val = [(tup[1], tup[0]) for tup in ordered_shap_list]
            model_output = (explainer_shap.expected_value + shap_values.sum()).round(4)  # list of probs

            print('ST model output anas', model_output)
            class_pred = np.argmax(abs(model_output))

            dizio = {'batch %s' % k: {'row %s' % k: {
                'class_names': class_names,
                'ST_prediction': pred,  # ST Predicted
                'SHAP_probs': model_output,
                'is ML correct': class_pred == pred,
                'value_ordered': feat_shap_val,  # (feature, shap_value) : ordered list
                'swapped': swap_shap,  # (feature, bool) : ordered list of swapped variables
                'SHAP_prediction': class_pred,  # xai prediction : class
                'shap_values': shap_values
            }}}
            ret_st.append(dizio)

            ############################### LIME ################################
            start_time_lime = time.time()
            exp_lime = explainer_lime.explain_instance(diz['X_test'][0],
                                                       diz['model'].predict,
                                                       num_samples=100,
                                                       num_features=len(all_cols),
                                                       distance_metric='euclidean')
            tot_lime = time.time() - start_time_lime
            time_lime.append(tot_lime)
            #st_time_lime1 = f"- ST - LIME Total time {filename}: {(time.time() - start_time_lime) / 60} minutes"

            lime_prediction = exp_lime.local_pred
            exp_lime.as_pyplot_figure().tight_layout()
            plt.text(0.3, 0.7, f' D3 Local LIME row {str(k)}')
            plt.savefig('images/' + f' ST Local LIME row {str(k)} dataset {filename}')
            exp_lime.save_to_file(f'ST_lime_row_{str(k)}_dataset_{filename}.html') #save_to_file('lime.html')
            big_lime_list = exp_lime.as_list()  # list of tuples (representation, weight),
            ord_lime = sorted(big_lime_list, key=lambda x: abs(x[1]), reverse=True)

            variables = []  # (name, real value)
            f_weight = []   # (feature, weight)
            swap = []       # (feature, bool)

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


            lime_diz = {'batch %s' % k: {'row %s' % k: {
                'class_names': class_names,
                'ST_prediction': pred,  # ST prediction
                'LIME_LOCAL_prediction': lime_prediction,  # xai prediction
                'LIME_GLOBAL_prediction': exp_lime.predicted_value,  # xai global prediction
                'value_ordered': f_weight,  # (feature, lime_weight)
                'feature_value': variables,  # (feature, real value)
                'swapped': swap  # (feature, bool)

            }}}

            lime_res_st.append(lime_diz)

            ################### ANCHORS #########################################
            start_time_anch = time.time()
            class_list_str = ['0', '1']
            #pred_uno = class_list_str[explainer_anchor.predictor(diz['X_test'][k].reshape(1, -1))[0]]
            #print('ALIBI PREDICTION ANAS', pred_uno)

            exp_anchor = explainer_anchor.explain(diz['X_test'][k],
                                                  threshold=0.90,
                                                  beam_size=len(all_cols),
                                                  coverage_samples=1000)
            print('ALIBI EXPLANATION ANAS', exp_anchor)

            # exp_anchor = explainer_anchor.explain_instance(diz['X_test'][0],
            #                                                diz['model'].predict,
            #                                                threshold=0.90,
            #                                                beam_size=len(all_cols))
            end_time_a = time.time() - start_time_anch
            time_anchor.append(end_time_a)

            rules = exp_anchor.anchor
            print('RULES', rules)
            precision = exp_anchor.precision
            coverage = exp_anchor.coverage

            #prediction_anch = diz['model'].predict(diz['X_test'].reshape(1, -1))[0]

            """
            # exp_anchor.show_in_notebook()
            # exp_anchor.examples(only_different_prediction = True)
            # print('esempi',exp_anchor.examples()) #np.ndarray

            rules = exp_anchor.names()
            precision = round(exp_anchor.precision(), 3)
            coverage = round(exp_anchor.coverage(), 3)

            '''
            print()
            print('anchor: %s' % (' AND '.join(exp_anchor.names())))
            print('precision: %.2f' % exp_anchor.precision())
            print('coverage: %.2f' % exp_anchor.coverage())

            # Get test examples where the anchora applies
            fit_anchor = np.where(np.all(diz['X_test'][i:, exp_anchor.features()] == diz['X_test'][i][exp_anchor.features()], axis=1))[0]
            print('Anchor test precision: %.2f' % (np.mean(diz['model'].predict(diz['X_test'][fit_anchor]) == diz['model'].predict(diz['X_test'][i].reshape(1, -1)))))
            print('Anchor test coverage: %.2f' % (fit_anchor.shape[0] / float(test_set.shape[0])))    
            print()'''
            """

            contrib = []
            swapped = []

            if len(rules) == 0:
                if len(contrib) > 0:
                    contrib.append(contrib[-1])
                else:
                    contrib.append(
                        'empty rule: all neighbors have the same label')  # al primo batch potrebbe essere vuoto


            else:
                for s in rules:                 # nel caso in cui ci siano più predicati
                    splittato = s.split(' ')    # splittato = [nswprice, >, 0.08], [0.3 <= feature <= 0.6]
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

                'ML_prediction': pred,
                #'Anchor_prediction': pred_uno,
                #'rule': ' AND '.join(exp_anchor.names()),
                'rule': ' AND '.join(exp_anchor.anchor),
                'precision': precision,
                'coverage': coverage,
                'swapped': swapped,
                'value_ordered': contrib,
            }}}

            anchor_res_st.append(diz_anchors)

            k += 1

        # ST FILES
        with open('results/' + 'ST_SHAP_%s.json' % filename, 'w', encoding='utf-8') as f:
            json.dump(ret_st, f, cls=NumpyEncoder)

        with open('results/' + 'ST_LIME_%s.json' % filename, 'w', encoding='utf-8') as f11:
            json.dump(lime_res_st, f11, cls=NumpyEncoder)

        with open('results/' + 'ST_ANCHOR_REGRESSION_%s.json' % filename, 'w', encoding='utf-8') as f12:
            json.dump(anchor_res_st, f12, cls=NumpyEncoder)

        with open('other_files/' + f"ST - SHAP - Total time {filename}", 'w', encoding='utf-8') as t9:
            json.dump(time_shap, t9, cls=NumpyEncoder)

        with open('other_files/' + f"ST - LIME - Total time {filename}", 'w', encoding='utf-8') as t10:
            json.dump(time_lime, t10, cls=NumpyEncoder)

        with open('other_files/' + f"ST - ANCHORS - Total time {filename}", 'w', encoding='utf-8') as t11:
            json.dump(time_anchor, t11, cls=NumpyEncoder)

        mean_time_shap = np.mean(time_shap)
        mean_time_lime = np.mean(time_lime)
        mean_time_anchor = np.mean(time_anchor)
        index = ['mean_time_shap', 'mean_time_lime', 'mean_time_anchor']
        means = [mean_time_shap, mean_time_lime, mean_time_anchor]
        to_export = pd.DataFrame(means, columns=['mean time'], index=index)
        to_export.to_excel(f'other_files/ST {filename}.xlsx')

        f.close()
        f11.close()
        f12.close()
        t9.close()
        t10.close()
        t11.close()

        #return ret, lime_res, anchor_res

print(f"xai_anas.py Total time: {(time.time() - first_time) / 60} minutes")
# tot_time_anas = f"st_anas.py Total time: {(time.time() - first_time) / 60} minutes"
# with open('other_files/' + f"ST - ANCHOR - Total time: {tot_time_anas}", 'w', encoding='utf-8') as t8:
#     json.dump(tot_time_anas, t8, cls=NumpyEncoder)
print('END xai_anas')