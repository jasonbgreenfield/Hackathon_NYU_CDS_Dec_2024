"""
Name: __main__.py
Date: Dec 3, 2024
Author: Jason Greenfield, Ben Guinaudeau, Edwin Kamau, Yuchunji Lu
Purpose: run inference for CDS Hackathon task: use traditional ml methods for health data
Data In: hackathon syntehtic data
Machine: NYU hpc
"""
import argparse
import os

import pandas as pd
import numpy as np
import re
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


TESTING = False


def wrangle_data():
    # Columns to remove
    col_to_remove = ["patientid", "sex", "agecategory", "heightinmeters","weightinkilograms","hadheartattack"]

    # Remaining columns
    remaining = ["label", "state", "raceethnicitycategory", "tetanuslast10tdap", "gender", "generalhealth", "age", "bmi", "hadangina", "hadstroke", "hadasthma", "hadskincancer", "hadcopd", "haddepressivedisorder", "hadkidneydisease", "hadarthritis", "haddiabetes", "deaforhardofhearing", "blindorvisiondifficulty", "difficultyconcentrating", "difficultywalking", "difficultydressingbathing", "difficultyerrands", "smokerstatus", "ecigaretteusage", "chestscan", "alcoholdrinkers", "hivtesting", "fluvaxlast12", "pneumovaxever", "highrisklastyear", "covidpos"]

    # read in data
    x = pd.read_csv('/scratch/jg7477/cds_hackathon_20241203/ha_train_set/inputs.csv')
    y = pd.read_csv('/scratch/jg7477/cds_hackathon_20241203/ha_train_set/labels.csv')

    # Perform operations on DataFrame
    dt = pd.merge(x, y, how="left")  # Left join
    dt.columns = dt.columns.str.lower()  # Clean column names
    # Mutate
    dt['label'] = np.where(dt['hadheartattack'] == 1, "Heart Attack", "No heart Attack")
    dt['gender'] = (dt['sex'] == "Female").astype(int)
    # Recode `general_health`
    general_health_map = {"Excellent": 5,"Very good": 4,"Good": 3,"Fair": 2,"Poor": 1}
    dt['generalhealth'] = dt['generalhealth'].map(general_health_map)
    # Recode `had_diabetes`
    had_diabetes_map = {"Yes": 4,"No, pre-diabetes or borderline diabetes": 3,"Yes, but only during pregnancy (female)": 2,"No": 1}
    dt['haddiabetes'] = dt['haddiabetes'].map(had_diabetes_map)
    # Recode `smoker_status`
    smoker_status_map = {"Current smoker - now smokes every day": 4,"Current smoker - now smokes some days": 3,"Former smoker": 2,"Never smoked": 1}
    dt['smokerstatus'] = dt['smokerstatus'].map(smoker_status_map)
    # Recode `e_cigarette_usage`
    e_cigarette_usage_map = {"Use them every day": 4,"Use them some days": 3,"Not at all (right now)": 2,"Never used e-cigarettes in my entire life": 1}
    dt['ecigaretteusage'] = dt['ecigaretteusage'].map(e_cigarette_usage_map)
    # Extract age from `age_category`
    dt['age'] = dt['agecategory'].str.extract(r"(\d+)").astype(float)
    # Drop unnecessary columns
    dt = dt.drop(columns=col_to_remove)
    # Reorder columns
    remaining_columns = [col for col in remaining if col in dt.columns]
    dt = dt[remaining_columns + [col for col in dt.columns if col not in remaining_columns]]
    # Convert label to binary
    dt['label'] = np.where(dt['label'] == "Heart Attack", 1, 0)
    # Convert categorical columns to numeric
    categorical_cols = dt.select_dtypes(include='object').columns
    dt[categorical_cols] = dt[categorical_cols].apply(lambda col: pd.factorize(col)[0] + 1)

    # Display DataFrame structure
    print(dt.info())

    # return wrangled data
    return dt


def run_inference(dt):
    ###
    # 0. PREP DATA
    ###
    # Split into training and testing sets
    np.random.seed(123)
    # fix labels
    dt['label'] = dt['label'].replace({1: 0, 2: 1})
    # Assume 'dt' is a pandas DataFrame
    dt['train'] = np.random.rand(len(dt)) < 0.8
    train = dt[dt['train']].copy()
    test = dt[~dt['train']].copy()
    # Prepare data for XGBoost
    train_matrix = xgb.DMatrix(data=train.drop(columns=['label', 'train']), label=train['label'])
    test_matrix = xgb.DMatrix(data=test.drop(columns=['label', 'train']), label=test['label'])

    ###
    # 1. TRAIN MODEL
    ###
    # Train the XGBoost model
    xgb_model = xgb.train(
        {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'eta': 0.3
        },
        train_matrix,
        num_boost_round=100,
        evals=[(train_matrix, 'train'), (test_matrix, 'test')],
        verbose_eval=True
    )

    ###
    # 2. RUN INFERENCE
    ###
    # run inference on model
    predictions = xgb_model.predict(test_matrix)
    # convert prediction for probability of heart attack to binary cutoff
    class_labels = (predictions > 0.3).astype(int)
    # get true labels
    true_labels = test_matrix.get_label()

    ###
    # 3. EVALUATE PREDICTIONS
    ###
    # Compute accuracy
    accuracy = accuracy_score(true_labels, class_labels)

    # Compute precision
    precision = precision_score(true_labels, class_labels)

    # Compute recall
    recall = recall_score(true_labels, class_labels)

    # Compute F1-score
    f1 = f1_score(true_labels, class_labels)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, class_labels)

    # Compute ROC-AUC score (requires probability predictions, not class labels)
    # Assuming `predictions` is the probability output from `xgb.predict()`
    auc = roc_auc_score(true_labels, predictions)

    # Print metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)
    print("ROC-AUC Score:", auc)


def main_testing():
    print('wrangling data...')
    df = wrangle_data()
    print('running inference...')
    run_inference(df)
    print('done')


def main(test_set_dir, results_dir):
    # Load test set data.
    input_df = pd.read_csv(os.path.join(test_set_dir, "inputs.csv"))

    # ---------------------------------
    # START PROCESSING TEST SET INPUTS
    # Beep boop bop you should do something with test inputs unlike this script.

    patients = list(input_df.PatientID)
    output_df = pd.DataFrame(columns=["PatientID", "HadHeartAttack"])
    output_df["PatientID"] = patients
    heart_attack_percent = 0
    had_heart_attack = np.random.random(len(patients)) < heart_attack_percent
    output_df["HadHeartAttack"] = had_heart_attack

    # END PROCESSING TEST SET INPUTS
    # ---------------------------------

    # NOTE: name "results.csv" is a must.
    output_df.to_csv(os.path.join(results_dir, "results.csv"), index=False)


if __name__ == "__main__":
    if TESTING:
        pass
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--bth_test_set",
            type=str,
            required=True
        )
        parser.add_argument(
            "--bth_results",
            type=str,
            required=True
        )

        args = parser.parse_args()
        main(args.bth_test_set, args.bth_results)

