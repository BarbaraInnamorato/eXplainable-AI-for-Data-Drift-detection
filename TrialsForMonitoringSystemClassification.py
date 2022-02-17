import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#read file
foldername="data/weather/"
df_labels = pd.read_csv(foldername + "NEweather_class.csv", header=None)

y = df_labels.values.flatten()  # numpy ndarray
# y.columns = ['PRCP']

df_data = pd.read_csv(foldername + "NEweather_data.csv", header=None)

df = df_data.copy()
df['y'] = y
df.columns = ['TEMP', 'DEWP', 'SLP', 'VISIB', 'WDSP', 'MXSPD', 'MAX', 'MIN', 'PRC']

X = df.iloc[:, :-1]
label = np.unique(df.iloc[:, -1:])  # 0,1
le = LabelEncoder()
le.fit(label)
y = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=472)


def plot_roc(y_test, y_pred, classes, name):
    # Compute ROC curve and ROC area for each class
    true = np.array(y_test).reshape(-1,1)
    pred = np.array(y_pred).reshape(-1,1)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], threshold1 = roc_curve(true, pred)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class (class 1)
    plt.figure()
    lw = 2
    plt.plot(
        fpr[0],
        tpr[0],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc[0],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver operating characteristic example {name}")
    plt.legend(loc="lower right")
    plt.savefig(f'ROC_{name}')

    return fpr, tpr

def run_exps(X_train, y_train, X_test, y_test):
    models = [
              ('LogReg', LogisticRegression(solver = 'liblinear')),
              ('RF', RandomForestClassifier()),
              ('SVM', make_pipeline(StandardScaler(), SVC())),
              ('XGB', XGBClassifier(label_encoder = False))
            ]

    results = []
    for name, model in models:
        clf = model.fit(X_train, np.array(y_train).reshape(1,-1)[0])
        y_pred = clf.predict(X_test)
        classes = np.unique(y_pred)
        print('-------',name)

        plot_roc(y_test, y_pred, classes, name)



        diz_res = {
            'model': name,
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'f1-score': metrics.f1_score(y_test, y_pred),
            'AUC': AUC(y_test, y_pred),
            'classification_report':classification_report(y_test, y_pred, labels=classes),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
        }


        # if name != 'LogReg' and name != 'SVM':
        #     feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
        #     feat_importances.nlargest(20).plot(kind='barh')
        #     diz_res['feature_importance'] = feat_importances

        results.append(diz_res)
    print(results)
    df = pd.DataFrame(results)
    df.to_excel('classificationTrials.xlsx')
    return results


run_exps(X_train, y_train, X_test, y_test)
