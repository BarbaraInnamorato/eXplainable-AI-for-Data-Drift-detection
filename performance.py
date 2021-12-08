import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import os
matplotlib.use('Agg')

if not os.path.exists('performances'):
    os.mkdir('performances')


def precision_k(predicted, actual, k):
    predicted_selected = predicted[:k]
    return len(set(predicted_selected) & set(actual)) / k


def recall_k(predicted, actual, k):
    predicted_selected = predicted[:k]
    return len(set(predicted_selected) & set(actual)) / len(actual)


def get_actual(values):
    actual = []
    for k, r in values:
        if r:
            actual.append(k)
    return actual
"""
- 'swapped': [  ['date', True], 
                ['day', True], 
                ['nswdemand', False], 
                ['vicdemand', False], 
                ['nswprice', True], 
                ['vicprice', False], 
                ['period', False], 
                ['transfer', True]]
la lista swapped è ordinata

- ACTUAL ['nswdemand', 'vicdemand', 'period', 'date'] sono i true in swapped --> stesso ordine di swapped
- SET ACTUAL {'vicdemand', 'nswdemand', 'period', 'date'}

- predicted ['date', 'day', 'nswdemand', 'vicdemand', 'nswprice', 'vicprice', 'period', 'transfer'] 
    --> stesso ordine di swapped
    --> in ordine di importanza in base allo xai method
    
parto dal primo elemento in predicted, vedo se è in actual
    se lo è --> 1
    se non lo è --> 0
considero il secondo predicted e vedo se è in actual 
--- a un certo punto potrei dire, ad esempio, 
"considerando 3 predictor, in actual ce nè uno
considerando 4 predictor, in actual ce ne sono 2
ecc... 

"""


def read_files():
    path_to_json = 'results/'
    directory = r'C:\Users\binnamorato\PycharmProjects\TESI_BARBARA\results'
    folder = [pos_json for pos_json in os.listdir(directory) if pos_json.endswith('.json')]

    for index, js in enumerate(folder):
        with open(os.path.join(path_to_json, js)) as json_file:
            data_dict_o = json.load(json_file)
            print('***data_dict_o', data_dict_o)
            print('****data_dict_o[1]', data_dict_o[1])

            result_name = "performance_%s" % js
            print('-------------name', result_name)
            columns = list(data_dict_o[1].values())[0]
            n = len(columns)
            k_range = list(range(1, n))


            data_dict = dict()
            i = 0
            for e in data_dict_o[2:]:               # {'batch 22': {'row 19' ecc...
                for b in e.values():                    # {'row 19': {'class_names': ecc...
                    for r in b.values():                    # {'class_names': [0, 1] ecc...
                        data_dict[i] = r
                        i += 1

            #print('data dict', data_dict)
            data = {}
            for key, v in data_dict.items():
                if 'Anchor_prediction' in v.keys() and v['swapped'] != []:
                    predicted = v['value_ordered']
                else:
                    predicted = [t[0] for t in v['value_ordered']]
                #predicted = [t[0] for t in v['value_ordered']]
                actual = get_actual(v['swapped'])

                if len(predicted) == 0 or len(actual) == 0: #succede con ANCHORS
                    #break
                    raise Exception('one list between predicted and actual is empty')
                else:
                    resulting_dict = {'predicted': predicted, 'actual': actual}
                    for k in k_range:
                        resulting_dict[f"P@{k}"] = precision_k(predicted, actual, k)
                        resulting_dict[f"R@{k}"] = recall_k(predicted, actual, k)
                    data[key] = resulting_dict

                    data_df = pd.DataFrame.from_dict(data, orient="index")
                    data_df['result'] = 0
                    performance_df = data_df.groupby('result').agg(['mean', 'std']).T
                    performance_df.to_excel(f'performances/{result_name[:-5]}.xlsx')


                    # PLOT
                    p_low = []
                    p_high = []
                    p_value = []
                    for k in k_range:
                        m = performance_df.loc[f"P@{k}"].loc['mean'].values[0]
                        s = performance_df.loc[f"P@{k}"].loc['std'].values[0]
                        p_value.append(m)
                        p_low.append(max(0, m-s))
                        p_high.append(min(1, m+s))


                    p_low = np.array(p_low)
                    p_high = np.array(p_high)
                    p_value = np.array(p_value)

                    r_low = []
                    r_high = []
                    r_value = []
                    for k in k_range:
                        m = performance_df.loc[f"R@{k}"].loc['mean'].values[0]
                        s = performance_df.loc[f"R@{k}"].loc['std'].values[0]
                        r_value.append(m)
                        r_low.append(max(0, m-s))
                        r_high.append(min(1, m+s))


                    r_low = np.array(r_low)
                    r_high = np.array(r_high)
                    r_value = np.array(r_value)

                    fig, ax = plt.subplots(figsize=(16, 8))
                    x_values = np.array(k_range)
                    p_color = '#1C325B'
                    r_color = '#DA291C'

                    ax.fill_between(
                                x_values, p_high, p_low,
                                interpolate=True, color=p_color, alpha=0.25,
                                label="Precision Range"
                                )

                    ax.fill_between(
                                x_values, r_high, r_low,
                                interpolate=True, color=r_color, alpha=0.25,
                                label="Recall Range"
                                )
                    ax.plot(x_values, p_value, color=p_color, label="Precision", lw=3)
                    ax.plot(x_values, r_value, color=r_color, label="Recall", lw=3)
                    ax.legend()
                    plt.title('%s'%result_name[:-5])
                    plt.tight_layout()
                    plt.savefig(r'performances/ %s.png'%result_name[:-5])
                    plt.close()

                    # chiudere il file
                    json_file.close()
                    plt.close('all')

