from sklearn.inspection import permutation_importance
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def compute_pfi(rfc, to_export, all_cols, filename):
    """
    The n_repeats parameter sets the number of times a feature is randomly shuffled and returns
    a sample of feature importances"""

    perm_importance = permutation_importance(rfc, to_export['X_test_post'], to_export['y_test_post'], n_repeats=10, random_state=42, n_jobs=2)
    perm_zip = list(zip(all_cols, perm_importance['importances_mean']))
    perm_sorted = sorted(perm_zip, key=lambda x: x[1], reverse=True)
    print('perm imp 2 (held out) ', perm_sorted)
    plt.figure()
    x_val = [t[0] for t in perm_sorted]
    y_val = [t[1] for t in perm_sorted]
    plt.barh(x_val, y_val, color='orange')
    plt.xlabel("Permutation Importance")
    plt.title('RF Permutation Feature Importance')
    plt.tight_layout()
    plt.savefig('images/' + f'RF Permutation Feature Importance {filename}')



