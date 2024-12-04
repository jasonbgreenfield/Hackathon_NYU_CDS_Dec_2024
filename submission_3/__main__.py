"""
Name: __main__.py
Date: Dec 3, 2024
Author: Jason Greenfield, Ben Guinaudeau, Edwin Kamau, Yucheng Lu
Purpose: run inference for CDS Hackathon task: use traditional ml methods for health data
Data In: hackathon syntehtic data
Machine: NYU hpc
"""
import argparse
import os
import pickle

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def wrangle_data(dt):
    # Columns to remove
    col_to_remove = ["patientid", "sex", "agecategory", "heightinmeters", "weightinkilograms", "hadheartattack"]

    # Remaining columns
    remaining = ["label", "state", "raceethnicitycategory", "tetanuslast10tdap", "gender", "generalhealth", "age",
                 "bmi", "hadangina", "hadstroke", "hadasthma", "hadskincancer", "hadcopd", "haddepressivedisorder",
                 "hadkidneydisease", "hadarthritis", "haddiabetes", "deaforhardofhearing", "blindorvisiondifficulty",
                 "difficultyconcentrating", "difficultywalking", "difficultydressingbathing", "difficultyerrands",
                 "smokerstatus", "ecigaretteusage", "chestscan", "alcoholdrinkers", "hivtesting", "fluvaxlast12",
                 "pneumovaxever", "highrisklastyear", "covidpos"]

    dt.columns = dt.columns.str.lower()  # Clean column names
    # Mutate
    dt['label'] = np.where(dt['hadheartattack'] == 1, "Heart Attack", "No heart Attack")
    dt['gender'] = (dt['sex'] == "Female").astype(int)
    # Recode `general_health`
    general_health_map = {"Excellent": 5, "Very good": 4, "Good": 3, "Fair": 2, "Poor": 1}
    dt['generalhealth'] = dt['generalhealth'].map(general_health_map)
    # Recode `had_diabetes`
    had_diabetes_map = {"Yes": 4, "No, pre-diabetes or borderline diabetes": 3,
                        "Yes, but only during pregnancy (female)": 2, "No": 1}
    dt['haddiabetes'] = dt['haddiabetes'].map(had_diabetes_map)
    # Recode `smoker_status`
    smoker_status_map = {"Current smoker - now smokes every day": 4, "Current smoker - now smokes some days": 3,
                         "Former smoker": 2, "Never smoked": 1}
    dt['smokerstatus'] = dt['smokerstatus'].map(smoker_status_map)
    # Recode `e_cigarette_usage`
    e_cigarette_usage_map = {"Use them every day": 4, "Use them some days": 3, "Not at all (right now)": 2,
                             "Never used e-cigarettes in my entire life": 1}
    dt['ecigaretteusage'] = dt['ecigaretteusage'].map(e_cigarette_usage_map)
    # Extract age from `age_category`
    dt['age'] = dt['agecategory'].str.extract(r"(\d+)").astype(float)
    # Drop unnecessary columns
    dt = dt.drop(columns=col_to_remove)
    # Reorder columns
    remaining_columns = [col for col in remaining if col in dt.columns]
    dt = dt[remaining_columns + [col for col in dt.columns if col not in remaining_columns]]
    # Convert categorical columns to numeric
    categorical_cols = dt.select_dtypes(include='object').columns
    dt[categorical_cols] = dt[categorical_cols].apply(lambda col: pd.factorize(col)[0] + 1)

    # Display DataFrame structure
    print(dt.info())

    # return wrangled data
    return dt


def train_model():
    # read in data
    x = pd.read_csv('/scratch/jg7477/cds_hackathon_20241203/ha_train_set/inputs.csv')
    dt = wrangle_data(x)

    # Select relevant columns
    dt = dt[[
        "label", "general_health", "chest_scan", "gender", "age", "had_angina",
        "chest_scan", "had_diabetes", "difficulty_walking", "had_stroke", "smoker_status"
    ]]
    # Separate features and target for training
    X_train = dt.drop(columns=['label'])
    y_train = dt['label']
    # Fit logistic regression model
    model = LogisticRegression(solver='liblinear')  # 'liblinear' is good for small datasets
    model.fit(X_train, y_train)

    # Save the model to a local file using pickle
    with open('/scratch/jg7477/cds_hackathon_20241203/logistic_regression_final_model_20241203.pkl', 'wb') as f:
        pickle.dump(model, f)


def main(test_set_dir, results_dir):
    # Load test set data.
    input_df = pd.read_csv(os.path.join(test_set_dir, "inputs.csv"))

    # ---------------------------------
    # START PROCESSING TEST SET INPUTS

    # Load the saved model from the file
    with open('logistic_regression_final_model_20241203.pkl', 'rb') as f:
        model = pickle.load(f)

    # generate predictions
    output_df = model.predict_proba(input_df)[:, 1]

    # END PROCESSING TEST SET INPUTS
    # ---------------------------------

    # NOTE: name "results.csv" is a must.
    output_df.to_csv(os.path.join(results_dir, "results.csv"), index=False)


if __name__ == "__main__":
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
