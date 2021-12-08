import warnings
warnings.filterwarnings("ignore")

import multiprocessing as mp
print('Number of processor', mp.cpu_count())

# Dataset loading
from datasetloader.load_dataset import load_stream

# Drift Methods
from studentteacher.student_teacher import *
from d3.D3 import *

# XAI techniques
import XAI
import xai_anas
import RF_xai

# Utilities
from progress.bar import IncrementalBar
import time
import os

# XAI Performace computation
import performance

# Create directories
if not os.path.exists('results'):
    os.mkdir('results')

if not os.path.exists('images'):
    os.mkdir('images')

if not os.path.exists('other_files'):
    os.mkdir('other_files') # for anchor files and RF metrics


# Setup
#models = ['d3', 'student-teacher']
models = ['d3']


n_repetitions = 1  # se lascio 1 poi devo togliere tutti i for

print('START')
print('-----')
start_time = time.time()


def faicose_un_dataset(dataset_name):

    print("----------Starting with dataset: {}".format(dataset_name))
    streams = []
    real_drift_points = []
    drift_point = 0

    # Drift Injection and Stream Creation
    for i in range(n_repetitions):
        dstream, drifted_rows, drift_point, drift_cols = load_stream(dataset_name, shuffle=True)
        streams.append(dstream)
        real_drift_points.append(drift_point)


    # Training Phase
    if dataset_name in ['anas']:
        teacher = Teacher('RandomForestRegressor')
        student = Student('RandomForestRegressor')
    else:
        teacher = Teacher('RandomForestClassifier')
        student = Student('RandomForestClassifier')

    train_results = []
    bar = IncrementalBar('Training Phase', max=len(streams))
    for s in streams:
        bar.next()
        if 'student-teacher' in models:
            print(' - Fitting ST')
            train_results.append(teacher_student_train(teacher, student, s, fit=True))
        else:
            print(' -Fitting %s' % (models[0]))
            train_results.append(teacher_student_train(teacher, student, s, fit=False))
    bar.finish()

    cols_to_print = [dstream.feature_names[x] for x in drift_cols]
    cl = dstream.feature_names

    # Detection Phase
    inf_results = {m: [] for m in models}
    anas_results = {m: [] for m in models}

    inference_functions = {
        'd3': d3_inference(drift_point, train_results),
        #'student-teacher': teacher_student_inference(drift_point,train_results)
        }

    ii = 1
    for idx, r in enumerate(train_results):
        print('r', r)
        r['drift_point'] = real_drift_points[idx]
        print("Iteration {}".format(ii))

        for m in models:
            print('model fitting',m)
            if not dataset_name in ['anas']:
                inf_results[m].append(inference_functions[m])
            else:
                anas_results[m].append(inference_functions[m])
        ii += 1

    print('Swapped columns for drift injection are', cols_to_print)
    print()

    # data for xai
    if dataset_name in ['anas']:
        #anas_st = anas_results['student-teacher'][0]
        anas_d3 = anas_results['d3'][0]
        XAI.d3_xai(anas_d3, cols_to_print, cl, dataset_name)
        #xai_anas.st_xai(anas_st, cols_to_print, cl, dataset_name)

    else:
        #st = inf_results['student-teacher'][0]
        d3 = inf_results['d3'][0]
        XAI.d3_xai(d3, cols_to_print, cl, dataset_name)
        #XAI.st_xai(st, cols_to_print, cl, dataset_name)


    """
    # Monitoring data - PERFORM RANDOM FOREST (REGRESSION/CLASSIFICATION)
    for idx, s in enumerate(streams):
        n_train = train_results[idx]['n_train']
        s.restart()

        X_train, y_train = s.next_sample(n_train)
        X_test_pre, y_test_pre = s.next_sample(real_drift_points[0] - n_train)
        X_test_post, y_test_post = s.next_sample(s.data.shape[0] - real_drift_points[0] )

        to_export = {'X_train':X_train,
                     'y_train': y_train,
                     'X_test_pre': X_test_pre,
                     'y_test_pre': y_test_pre,
                     'X_test_post': X_test_post,
                     'y_test_post': y_test_post
                      }
        if dataset_name in ['anas']:
            print('----------RANDOM FOREST %s'%dataset_name)
            RF_xai.rf_regression(to_export, cols_to_print, cl, dataset_name)
        else:
            print('----------RANDOM FOREST %s'%dataset_name)
            RF_xai.rf_classification(to_export, cols_to_print, cl, dataset_name)"""


def execute_main():

    print("Starting 'execute_main'")
    # creating processes
    #p1 = mp.Process(target=faicose_un_dataset, args=('electricity',))
    p2 = mp.Process(target=faicose_un_dataset, args=('electricity',))
    """
    p3 = mp.Process(target=faicose_un_dataset, args=('weather',))
    p4 = mp.Process(target=faicose_un_dataset, args=('forestcover',))"""

    # starting processes
    #print(p1.start())
    print(p2.start())
    """
    print(p3.start())
    print(p4.start())
    """

    # process IDs
    #print("ID of process p1: {}".format(p1.pid))
    print("ID of process p2: {}".format(p2.pid))
    """
    print("ID of process p3: {}".format(p3.pid))
    print("ID of process p4: {}".format(p4.pid))
    """

    # wait until processes are finished
    #p1.join()
    p2.join()
    """    
    p3.join()
    p4.join()
    """

    # all processes finished
    print("All processes finished execution!")

    # check if processes are alive
    #print("Process p1 is alive: {}".format(p1.is_alive()))
    print("Process p2 is alive: {}".format(p2.is_alive()))
    """
    print("Process p3 is alive: {}".format(p3.is_alive()))
    print("Process p4 is alive: {}".format(p4.is_alive()))
    """

    # Performances Computation (outside the for: takes files from results folder)
    performance.read_files()

    print(f"Total time: {(time.time() - start_time) / 60} minutes")
    print('---')
    print('END')

if __name__ == "__main__":
    execute_main()


