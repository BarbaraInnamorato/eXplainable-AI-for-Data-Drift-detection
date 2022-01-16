import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import lime
import lime.lime_tabular
from lime import submodular_pick

if not os.path.exists('sp_lime'):
    os.mkdir('sp_lime')




def sp_lime(data_for_xai, all_cols, filename):
    '''
    Function for submodular pick with LIME
    D3 ONLY !!

    '''
    for diz in data_for_xai[-1:]:
        class_names = np.unique(diz['y_train'])

        explainer_lime = lime.lime_tabular.LimeTabularExplainer(diz['X_train'],
                                                                mode='classification',
                                                                feature_names=all_cols,
                                                                feature_selection='none',
                                                                discretize_continuous=True,
                                                                discretizer='quartile',
                                                                #verbose=True
                                                                )

        sp_obj = submodular_pick.SubmodularPick(data=diz['X_test'],
                                                explainer=explainer_lime,
                                                num_features=len(all_cols),
                                                predict_fn=diz['model'].predict_proba,
                                                num_exps_desired=10,
                                                sample_size=20,
                                                top_labels=len(class_names)
                                                )
        # Plot the 10 explanations
        #plt.figure(figsize=(16, 5))
        #[exp.as_pyplot_figure().savefig('images/' + f'SP_LIME_{filename}') for exp in
         #sp_obj.sp_explanations]

        # empty= []
        for exp in sp_obj.sp_explanations:
            exp.as_pyplot_figure() #.tight_layout()
            plt.savefig('images/' + f'SP_LIME_D3{filename}', bbox_inches='tight')

        W = pd.DataFrame([dict(this.as_list()) for this in sp_obj.explanations])
        W.to_excel(f'sp_lime/PRIMO_SP_LIME_D3_{filename}.xlsx')

        # sp_explanations: A list of explanation objects that has a high coverage
        # explanations: All the candidate explanations saved for potential future use.
        # to compute LIME COVERAGE: len(sp_obj.sp_explanations) / len(sp_obj.explanations)

        print(f'---------------  LIME COVERAGE {filename}')
        print(len(sp_obj.sp_explanations) / len(sp_obj.explanations))
        print()


        df = pd.DataFrame({})
        for this_label in range(len(class_names)):
            #print('this_label', this_label)
            dfl = []
            for i, exp in enumerate(sp_obj.sp_explanations):
                exp.as_pyplot_figure().tight_layout()
                plt.savefig('sp_lime/' +f'SP_LIME_global_D3{filename}_row_{str(i)}');
                l = exp.as_list(label=this_label)
                l.append(("exp number", i))
                dfl.append(dict(l))
                #print('dfl', dfl)
            # dftest=pd.DataFrame(dfl)
            df = df.append(
                pd.DataFrame(dfl, index=[class_names[this_label] for i in range(len(sp_obj.sp_explanations))]))
            df.to_excel(f'sp_lime/SECONDO_SP_LIME_D3_{filename}.xlsx')


################################################

def st_sp_lime(data_for_xai, all_cols, filename):
    '''
    Function for submodular pick with LIME
    '''
    for diz in data_for_xai:
        print(diz)
        class_names = np.unique(diz['y_train'])

        mode = ''
        if filename == 'anas':
            mode += 'regression'
            pred_fn = diz['model'].predict,
        else:
            mode += 'classification'
            pred_fn = diz['model'].predict_proba,

        print('SP LIME MODE = ', mode)
        print('type xtrain', type(diz['X_train']), len(diz['X_train']))
        print('type xtest', type(diz['X_test']), len(diz['X_test']))
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(diz['X_train'],
                                                                mode=mode,
                                                                feature_names=all_cols,
                                                                feature_selection='none',
                                                                discretize_continuous=True,
                                                                discretizer='quartile',
                                                                #verbose=True
                                                                )

        sp_obj = submodular_pick.SubmodularPick(data=diz['X_test'],
                                                explainer=explainer_lime,
                                                num_features=len(all_cols),
                                                predict_fn=pred_fn,
                                                num_exps_desired=10,
                                                #sample_size=20,
                                                #top_labels= int(len(class_names))
                                                )
        # Plot the 10 explanations
        plt.figure(figsize=(16,5))


        for exp in sp_obj.sp_explanations:
            exp.as_pyplot_figure().tight_layout()
            plt.savefig('images/' + f'SP_LIME_ST_{filename}')

        #[exp.as_pyplot_figure().savefig('images/' + f'SP_LIME_ST{filename}') for exp in
        # sp_obj.sp_explanations]

        W = pd.DataFrame([dict(this.as_list()) for this in sp_obj.explanations])
        W.to_excel(f'sp_lime/PRIMO_SP_LIME_ST_{filename}.xlsx')
        # sp_explanations: A list of explanation objects that has a high coverage
        # explanations: All the candidate explanations saved for potential future use.
        # to compute LIME COVERAGE: len(sp_obj.sp_explanations) / len(sp_obj.explanations)

        print(f'---------------  LIME COVERAGE {filename}')
        print(len(sp_obj.sp_explanations) / len(sp_obj.explanations))
        print()
        lime_coverage = len(sp_obj.sp_explanations) / len(sp_obj.explanations)
        #lime_coverage.to_excel(f'sp_lime/COVERAGE_LIME_ST_{filename}.xlsx')


        df = pd.DataFrame({})
        for this_label in range(len(class_names)):
            #print('this_label', this_label)
            dfl = []
            for i, exp in enumerate(sp_obj.sp_explanations):
                exp.as_pyplot_figure().tight_layout()
                plt.savefig('sp_lime/' +f'SP_LIME_global_ST{filename}_row_{str(i)}');
                l = exp.as_list(label=this_label)
                l.append(("exp number", i))
                dfl.append(dict(l))
                #print('dfl', dfl)
            # dftest=pd.DataFrame(dfl)
            df = df.append(
                pd.DataFrame(dfl, index=[class_names[this_label] for i in range(len(sp_obj.sp_explanations))]))
            df.to_excel(f'sp_lime/SECONDO_SP_LIME_ST_{filename}.xlsx')



