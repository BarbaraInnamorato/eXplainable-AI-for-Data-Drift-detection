# eXplainable AI for data drift detection
# Master Thesis in Data Science @Unitn | BIP | 2020-2021


    pip install requirements

**FILE & PIPELINE DESCRIPTION**
Execution order

1 - **data folder** : contains the 4 datasets

2 - **datasetloader folder** : contains files for read datasets, create the streams, inject drift

3 - **d3 folder** : contains D3.py that includes
    - drift_detector function : returns the list of dictionaries for XAI
    - Class D3 : define w, rho, tau + driftCheck --> make trials with different values
    - d3_inference function : detect concept drift --> make trials with different Regressor

4 - **studentteacher folder** : contains student_teacher.py. 
    - Class Model : to define Student, Teacher for the training of all methods and datasets
    - teacher_student_train : training function always to be used --> returns train_results
    - teacher_student_inference : detect concept drift --> returns the list of dictionaries for XAI

5 - **XAI.py** --> SHAP, LIME, ANCHORS for electricity, forestcover, weather datasets
    *CLASSIFICATION ONLY*
    - d3_xai function to explain predictions of d3 algorithm
    - st_xai function to explain predictions of ST algorithm
    write json file saved into 'results' folder (not for anchors)

6 - **st_anas.py** --> SHAP, LIME, ANCHORS for anas dataset \
    *REGRESSION ONLY, ST ONLY*
    - st_xai function to explain predictions of ST algorithm
    write json file saved into 'results' folder (not for anchors)

7 - **RandomForest.py** : Monitoring with Random Forest - for all dataset. \
    - rf_regression function : for monitoring anas data
    - rf_classification function : for monitoring electricity, forestcover, weathere data
    write json files saved into 'results' folder (not for anchors)
    
8 - **Perm_importance.py** : Computes permutation feature importance of the Random Forest and makes plot \

9 - **performance.py** : compute precision and recall @k
    uses json files in the 'results' folder and for each one creates an excel file with
    precision and recall @k together and save the corresponding plot


    python3 main.py