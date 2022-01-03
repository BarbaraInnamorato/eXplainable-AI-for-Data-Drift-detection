from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor, \
    ExtraTreesRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Lasso
from skmultiflow.drift_detection.adwin import ADWIN

from progress.bar import IncrementalBar
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from datasetloader.load_dataset import *


class Model:

    def __init__(self, sel_model):
        models = {

            'ExtraTreeClassifier': ExtraTreesClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'NaiveBayes': GaussianNB(),
            'LogisticRegression': LogisticRegression(),
            'ExtraTreeRegressor': ExtraTreesRegressor(),
            'RandomForestRegressor': RandomForestRegressor(),
            'Lasso': Lasso()

        }
        self.ml_model = models[sel_model]
        self.regression_models = ['ExtraTreeRegressor', 'RandomForestRegressor', 'Lasso']
        if sel_model in self.regression_models:
            self.regression = True
        else:
            self.regression = False

    def fit(self, X, y):
        self.ml_model.fit(X, y)

    def predict(self, X):
        preds = self.ml_model.predict(X)
        return preds

    def predict_proba(self, X):
        probs = self.ml_model.predict_proba(X)
        return probs


class Teacher(Model):
    pass


class Student(Model):
    pass


def teacher_student_train(teacher, student, stream, fit=True, train_perc=0.6):
    n_train = int(train_perc * stream.data.shape[0])
    X_train, y_train = stream.next_sample(n_train)

    if fit:
        teacher.fit(X_train, y_train)
        y_hat_train = teacher.predict(X_train)

        student.fit(X_train, y_hat_train)

    train_results = {"Teacher": teacher, "Student": student, "n_train": n_train, "Stream": stream, "X_train": X_train,
                     "y_train": y_train}

    return train_results


def teacher_student_inference(drift_point, train_results):
    teacher = train_results[0]["Teacher"]
    student = train_results[0]["Student"]
    n_train = train_results[0]["n_train"]
    stream = train_results[0]["Stream"]
    X_train = train_results[0]["X_train"]
    y_train = train_results[0]['y_train']

    list_exp_dict = []
    error_list = []

    adwin = ADWIN()
    results = {'detected_drift_points': []}

    stream.restart()
    stream.next_sample(n_train)

    bar = IncrementalBar('ST_inference', max=stream.n_remaining_samples())
    i = n_train

    class_names = np.unique(y_train)
    while stream.has_more_samples():
        bar.next()

        Xi, yi = stream.next_sample()
        """
        y_hat_teacher = teacher.predict(Xi)
        y_hat_student = student.predict(Xi)

        student_error = int(y_hat_teacher != y_hat_student)
        """

        if teacher.regression is True:
            y_hat_teacher = teacher.predict(Xi)[0]
            y_hat_student = student.predict(Xi)[0]
            probs_teacher = y_hat_teacher
            class_idx = round(probs_teacher, 0)
            class_student = round(y_hat_student, 0)
            stud_probs = student.predict(Xi)[0]

        else:
            probs_teacher = teacher.predict_proba(Xi)[0]

            if len(probs_teacher) < 2:
                y_hat_teacher = probs_teacher[0]
                class_idx = np.argmax(probs_teacher)  # class teacher
                y_hat_student = student.predict_proba(Xi)[0][0]
                class_student = np.argmax(student.predict_proba(Xi)[0])
                stud_probs = list(student.predict_proba(Xi)[0])

            elif len(probs_teacher) == 2:

                y_hat_teacher = np.max(probs_teacher)
                class_idx = np.argmax(probs_teacher)
                y_hat_student = student.predict_proba(Xi)[0][class_idx]
                class_student = np.argmax(student.predict_proba(Xi)[0])
                stud_probs = list(student.predict_proba(Xi)[0])


            else:  # more than two classes
                y_hat_teacher = np.max(probs_teacher)
                class_idx = np.argmax(probs_teacher)
                y_hat_student = student.predict_proba(Xi)[0][class_idx]
                class_student = np.argmax(student.predict_proba(Xi)[0])
                stud_probs = list(student.predict_proba(Xi)[0])

        student_error = np.abs(y_hat_teacher - y_hat_student)
        adwin.add_element(student_error)
        error_list.append(student_error)

        if adwin.detected_change():
            if i > drift_point:
                print('-----------CONCEPT DRIFT ST -------------')
                print('Change detected in data: ' + str(Xi[i]) + ' - at index: ' + str(i))

                results['detected_drift_points'].append(i)

                exp_dict = {
                    'model': student,
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': Xi,
                    'y_test': yi,
                    'student_error': student_error, #tra 0 e 1
                    'drifted': True,
                    'indice_riga': i,

                    'y hat student': y_hat_student,
                    'y_hat_teacher': y_hat_teacher,
                    'class_names': class_names,
                    'probs_teacher': probs_teacher,
                    'class teacher': class_idx,
                    'probs_student': stud_probs,
                    'class_student': class_student
                }

                list_exp_dict.append(exp_dict)


        else:
            pass

        i += 1
    bar.finish()

    # return results
    return list_exp_dict
