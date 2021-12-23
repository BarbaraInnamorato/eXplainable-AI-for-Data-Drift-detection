from sklearn.inspection import permutation_importance
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def compute_pfi(rfc, to_export, all_cols, filename):
    # print('PERM IMPORTANCE - held out set')
    """
    The n_repeats parameter sets the number of times a feature is randomly shuffled and returns
    a sample of feature importances.

    """
    result = permutation_importance(rfc, to_export['X_test_post'], to_export['y_test_post'], n_repeats=10, random_state=42, n_jobs=2)
    print('results', result)
    sorted_idx = result.importances_mean.argsort()
    to_zip = zip(all_cols, sorted_idx)
    print('perm imp 1 ', to_zip)
    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=[all_cols[el] for el in sorted_idx])
    ax.set_title("Permutation Importances (test set)")
    plt.tight_layout()
    plt.savefig('images/' + 'RF Permutation Feature Importance BOXPLOT %s' % filename)


    """
    As an alternative, the permutation importances of rf are computed on a held out test set. 
    This shows that the low cardinality categorical feature, sex is the most important feature.
    
    Also note that both random features have very low importances (close to 0) as expected.
    """
    perm_importance = permutation_importance(rfc, to_export['X_test_post'], to_export['y_test_post'])
    perm_zip = list(zip(all_cols, perm_importance['importances_mean']))
    perm_sorted = sorted(perm_zip, key=lambda x: x[1])
    print('perm imp 2 (held out) ', perm_sorted)
    plt.figure(figsize=(12, 10))
    x_val = [t[0] for t in perm_sorted]
    y_val = [t[1] for t in perm_sorted]
    plt.barh(x_val, y_val, color='maroon')
    plt.xlabel("Permutation Importance")
    plt.title('RF Permutation Feature Importance')
    plt.tight_layout()
    plt.savefig('images/' + f'RF Permutation Feature Importance 2 {filename}')



