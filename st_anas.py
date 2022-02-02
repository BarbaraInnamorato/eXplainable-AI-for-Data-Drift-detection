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
import xlsxwriter

first_time = time.time() # returns the processor time

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

        student_error = []

        for diz in data_for_xai:
            print('---- ST diz------')
            student_error.append(diz['student_error'])
            train_set = pd.DataFrame(diz['X_train'], columns=all_cols)
            test_set = pd.DataFrame(diz['X_test'], columns=all_cols)

            # Setting explainers
            explainer_shap = shap.KernelExplainer(diz['model'].predict,
                                                  shap.sample(train_set, nsamples=100, random_state=90),
                                                  link='identity',
                                                  l1_reg=len(all_cols)
                                                  )

            explainer_lime = lime.lime_tabular.LimeTabularExplainer(diz['X_train'],
                                                                    mode='regression',
                                                                    feature_names=all_cols,
                                                                    feature_selection='none',
                                                                    discretize_continuous=True,
                                                                    verbose=True
                                                                    )
            predict_fn = lambda x: diz['model'].predict(x)
            explainer_anchor = AnchorTabular(predict_fn, all_cols)
            explainer_anchor.fit(diz['X_train'], disc_perc=(25, 50, 75))

            pred = diz['probs_student']

            # SHAP
            start_time_s = time.time()
            print('ST SHAP sto calcolando shap values')
            shap_values = explainer_shap.shap_values(test_set,  nsamples=100)  # test set è una riga
            #print('ST ANAS SHAP VALUES', shap_values)
            end_time_shap = (time.time() - start_time_s) / 60
            time_shap.append(end_time_shap)

            zipped = list(zip(shap_values[0], all_cols))
            ordered_shap_list = sorted(zipped, key=lambda x: abs(x[0]), reverse=True)

            # Get the force plot for each row
            shap.initjs()
            shap.plots.force(explainer_shap.expected_value, shap_values, test_set , feature_names=all_cols, matplotlib = True, show=False, text_rotation=6)#
            plt.savefig('html_images/'+ f'ST Local SHAP row {str(k)} dataset {filename}', bbox_inches='tight')

            swap_shap = [(tup[1], True if tup[1] in cols else False) for tup in ordered_shap_list]
            feat_shap_val = [(tup[1], tup[0]) for tup in ordered_shap_list]
            model_output = (explainer_shap.expected_value + shap_values.sum()).round(4)  # è uguale a ML prediction
            #print('ST model output anas', model_output)

            dizio = {'batch %s' % k: {'row %s' % k: {
                'student_error': diz['student_error'],
                'ST_prediction': pred,              # ST Prediction
                'drift': diz['drifted'],
                'value_ordered': feat_shap_val,     # (feature, shap_value) : ordered list
                'swapped': swap_shap,               # (feature, bool) : ordered list of swapped variables
                'shap_values': shap_values
            }}}
            ret_st.append(dizio)

            # LIME
            start_time_lime = time.time()
            exp_lime = explainer_lime.explain_instance(diz['X_test'][0],
                                                       diz['model'].predict,
                                                       num_samples=100,
                                                       num_features=len(all_cols),
                                                       distance_metric='euclidean'
                                                       )
            tot_lime = time.time() - start_time_lime
            time_lime.append(tot_lime)

            lime_prediction = exp_lime.local_pred
            exp_lime.as_pyplot_figure().tight_layout()
            plt.savefig('images/' + f'ST Local LIME row {str(k)} dataset {filename}')
            exp_lime.save_to_file('html_images/' + f'ST_lime_row_{str(k)}_dataset_{filename}.html') #save_to_file('lime.html')
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
                'student_error': diz['student_error'],
                'ST_prediction': pred,                                  # ST prediction
                'LIME_LOCAL_prediction': lime_prediction,               # xai prediction
                'LIME_GLOBAL_prediction': exp_lime.predicted_value,     # ML prediction (=Right)
                'value_ordered': f_weight,                              # (feature, lime_weight)
                'feature_value': variables,                             # (feature, real value)
                'swapped': swap                                         # (feature, bool)
            }}}

            lime_res_st.append(lime_diz)

            k += 1

        # ST FILES
        with open('results/' + 'ST_SHAP_%s.json' % filename, 'w', encoding='utf-8') as f:
            json.dump(ret_st, f, cls=NumpyEncoder)

        with open('results/' + 'ST_LIME_%s.json' % filename, 'w', encoding='utf-8') as f11:
            json.dump(lime_res_st, f11, cls=NumpyEncoder)

        mean_time_shap = np.mean(time_shap)
        mean_time_lime = np.mean(time_lime)
        mean_time_anchor = np.mean(time_anchor)
        index = ['mean_time_shap', 'mean_time_lime', 'mean_time_anchor']
        means = [mean_time_shap, mean_time_lime, mean_time_anchor]
        to_export = pd.DataFrame(means, columns=['mean time'], index=index)
        to_export.to_excel(f'other_files/ST_TIME_{filename}.xlsx')

        with xlsxwriter.Workbook(f'ST_StudentError_{filename}.xlsx') as workbook:  # generate file test.xlsx
            worksheet = workbook.add_worksheet()

            for row, data in enumerate(student_error):
                worksheet.write_row(row, 0, data)

        f.close()
        f11.close()

print(f"xai_anas.py Total time: {(time.time() - first_time) / 60} minutes")
print('END st_anas')

