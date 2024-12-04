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
import re
import time

from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from pytorch_tabnet.tab_model import TabNetClassifier
import torch


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
    print(dt['label'].value_counts())

    # return wrangled data
    return dt


def gradientboost_run_inference(df):
    # Balance the dataset (optional, depending on strategy)
    # Here, we'll use undersampling for simplicity
    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]

    df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)

    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    X = df_balanced.drop('label', axis=1)
    y = df_balanced['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(random_state=42))
    ])

    # Perform a grid search for hyperparameter tuning
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.1, 0.01],
        'model__max_depth': [3, 5],
    }

    grid_search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    print(f'time for grid search: {time.time() - start_time}')

    # Evaluate the best model
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

    """
    CONSOLE OUTPUT FROM 12/3/2024 17:49:54 TESTING 
    >>> print("Classification Report:\n", classification_report(y_test, y_pred))
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.80      0.80      0.80      2164
               1       0.79      0.79      0.79      2065
    
        accuracy                           0.80      4229
       macro avg       0.80      0.80      0.80      4229
    weighted avg       0.80      0.80      0.80      4229
    
    >>> print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    Confusion Matrix:
     [[1738  426]
     [ 428 1637]]
    >>> print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
    ROC AUC Score: 0.8870800195136797  
    """


def xgboost_smote(df):
    # Balance the dataset (optional, depending on strategy)
    # Here, we'll use undersampling for simplicity
    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]

    df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)

    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    X = df_balanced.drop('label', axis=1)
    y = df_balanced['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Resample the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train an XGBoost classifier
    xgb = XGBClassifier(random_state=42, scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]))
    xgb.fit(X_train_resampled, y_train_resampled)

    # Evaluate
    y_pred = xgb.predict(X_test)
    y_pred_proba = xgb.predict_proba(X_test)[:, 1]

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))


def multiple_models(df):
    # Balance the dataset (optional, depending on strategy)
    # Here, we'll use undersampling for simplicity
    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]

    df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)

    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    X = df_balanced.drop('label', axis=1)
    y = df_balanced['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models to test
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42)
    }

    # Dictionary to store results
    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

        # Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        confusion = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else "N/A"

        # Print evaluation
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"\n{name} Confusion Matrix:\n{confusion}")
        print(f"\n{name} ROC AUC Score: {roc_auc}")

        # Save results
        results[name] = {
            'classification_report': report,
            'confusion_matrix': confusion,
            'roc_auc_score': roc_auc
        }

    # Print summary of results
    print("\nSummary of Results:")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  - ROC AUC: {metrics['roc_auc_score']}")
        print(f"  - Precision (class 1): {metrics['classification_report']['1']['precision']}")
        print(f"  - Recall (class 1): {metrics['classification_report']['1']['recall']}")
        print(f"  - F1 Score (class 1): {metrics['classification_report']['1']['f1-score']}")
        print()


def models_test_4(df):
    # Train-test split
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Add polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    # Define base models
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                                 learning_rate=0.05, n_estimators=100, random_state=42)
    }

    # Define ensemble (voting classifier)
    ensemble = VotingClassifier(
        estimators=[
            ('lr', models['Logistic Regression']),
            ('rf', models['Random Forest']),
            ('gb', models['Gradient Boosting']),
            ('xgb', models['XGBoost']),
        ],
        voting='soft'  # Use probabilities for soft voting
    )

    # Perform cross-validation
    cv_results = {}
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        start_time = time.time()
        print(f"Evaluating {name} with cross-validation...")
        cv_score = cross_val_score(model, X_train_poly, y_train, cv=kf, scoring='roc_auc')
        cv_results[name] = np.mean(cv_score)
        print(f"{name} Mean AUC: {cv_results[name]:.4f}")
        print(f'Time for {name}: {time.time() - start_time}\n')

    # Evaluate ensemble
    print("Evaluating Ensemble with cross-validation...")
    ensemble_cv_score = cross_val_score(ensemble, X_train_poly, y_train, cv=kf, scoring='roc_auc')
    cv_results['Ensemble'] = np.mean(ensemble_cv_score)
    print(f"Ensemble Mean AUC: {cv_results['Ensemble']:.4f}")

    # Train and evaluate the ensemble on test set
    start_time = time.time()
    ensemble.fit(X_train_poly, y_train)
    y_pred = ensemble.predict(X_test_poly)
    y_pred_proba = ensemble.predict_proba(X_test_poly)[:, 1]
    print(f'Time for Ensemble: {time.time() - start_time}\n')

    print("\nFinal Test Evaluation for Ensemble:")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

    """
    OUTPUT FROM TESTING ON 12/3/2024 20:31:44 TESTING
    >>> models_test_4(df)
    Evaluating Logistic Regression with cross-validation...
    Logistic Regression Mean AUC: 0.8843
    Time for Logistic Regression: 15.956579446792603
    
    Evaluating Random Forest with cross-validation...
    Random Forest Mean AUC: 0.8622
    Time for Random Forest: 448.7936339378357
    
    Evaluating Gradient Boosting with cross-validation...
    Gradient Boosting Mean AUC: 0.8866
    Time for Gradient Boosting: 2236.970309495926
    
    Evaluating XGBoost with cross-validation...
    XGBoost Mean AUC: 0.8841
    Time for XGBoost: 50.94605112075806
    
    Evaluating Ensemble with cross-validation...
    Ensemble Mean AUC: 0.8878
    
    Final Test Evaluation for Ensemble:
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.97      0.96      0.97     35918
               1       0.43      0.54      0.48      2114
    
        accuracy                           0.94     38032
       macro avg       0.70      0.75      0.72     38032
    weighted avg       0.94      0.94      0.94     38032
    
    ROC AUC Score: 0.8888029171671014
    """


def model_testing_5_tabnet(df, balance=True):
    """
    Train and evaluate a TabNet model for binary classification with a balanced dataset.

    Returns:
    - model: Trained TabNetClassifier.
    - metrics: A dictionary containing ROC AUC and classification report.
    """
    if balance:
        # Balance the dataset (optional, depending on strategy)
        # Here, we'll use undersampling for simplicity
        df_majority = df[df.label == 0]
        df_minority = df[df.label == 1]

        df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)

        df_balanced = pd.concat([df_majority_downsampled, df_minority])

        X = df_balanced.drop('label', axis=1)
        y = df_balanced['label']

        # Split the data into train (80%) and eval + test (20%)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Split the eval + test data (X_temp, y_temp) into eval (50%) and test (50%) of the remaining 20%
        X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42,
                                                          stratify=y_temp)
    else:
        # Separate the features and labels
        X = df.drop('label', axis=1)
        y = df['label']

        # Split the data into train (80%) and eval + test (20%)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Split the eval + test data (X_temp, y_temp) into eval (50%) and test (50%) of the remaining 20%
        X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42,
                                                          stratify=y_temp)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to numpy arrays (required by TabNet)
    X_train_scaled, X_test_scaled, X_eval_scaled = np.array(X_train_scaled), np.array(X_test_scaled), np.array(X_eval)
    y_train, y_test, y_eval = np.array(y_train), np.array(y_test), np.array(y_eval)

    # Define TabNet model
    print('initializing TabNetClassifier...')
    model = TabNetClassifier(
        n_d=8,  # Width of decision prediction layer
        n_a=8,  # Width of attention embedding
        n_steps=3,  # Number of decision steps
        gamma=1.3,  # Relaxation parameter
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=1
    )

    # Train TabNet model
    print('fitting TabNetClassifier...')
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_eval_scaled, y_eval)],
        eval_metric=['auc'],
        max_epochs=50,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128
    )

    # Predict and evaluate
    print('getting predictions...')
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)

    metrics = {
        "ROC AUC": roc_auc,
        "Classification Report": report
    }

    print("\nMetrics:")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(report)

    return model, metrics


def main_testing():
    print('wrangling data...')
    df = wrangle_data()
    print('running inference...')
    gradientboost_run_inference(df)
    print('done')


def main(test_set_dir, results_dir):
    # Load test set data.
    input_df = pd.read_csv(os.path.join(test_set_dir, "inputs.csv"))

    # ---------------------------------
    # START PROCESSING TEST SET INPUTS
    # Beep boop bop you should do something with test inputs unlike this script.

    # patients = list(input_df.PatientID)
    # output_df = pd.DataFrame(columns=["PatientID", "HadHeartAttack"])
    # output_df["PatientID"] = patients
    # heart_attack_percent = 0
    # had_heart_attack = np.random.random(len(patients)) < heart_attack_percent
    # output_df["HadHeartAttack"] = had_heart_attack

    df = wrangle_data()

    # Balance the dataset (optional, depending on strategy)
    # Here, we'll use undersampling for simplicity
    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]

    df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)

    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    X = df_balanced.drop('label', axis=1)
    y = df_balanced['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(random_state=42))
    ])

    # Perform a grid search for hyperparameter tuning
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.1, 0.01],
        'model__max_depth': [3, 5],
    }

    grid_search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    print(f'time for grid search: {time.time() - start_time}')

    # Evaluate the best model
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

    ###
    # GENERATE PREDICTIONS FOR INPUT DATA
    ###
    output_df = best_model.predict(input_df)

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

