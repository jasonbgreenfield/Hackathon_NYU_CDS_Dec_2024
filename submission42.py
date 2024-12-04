"""
Name: __main__.py
Date: Dec 3, 2024
Author: Jason Greenfield, Ben Guinaudeau, Edwin Kamau, Yucheng Ly
Purpose: run inference for CDS Hackathon task: use traditional ml methods for health data
Data In: hackathon syntehtic data
Machine: NYU hpc
"""
import argparse
import os
import pickle

import pandas as pd
import numpy as np

from itertools import combinations
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb




def create_interactions(df, exclude_feature="PatientID", features=None, categorical_features=None, encoding_method='onehot', 
                       max_unique_thresh=10, drop_original=False):
    """
    Create pairwise interactions between features, handling both numerical and categorical variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the features
    exclude_feature : str, default="PatientID"
        Column name to exclude from interactions and preserve in output
    features : list, optional
        List of feature names to consider for interactions. If None, uses all columns
    categorical_features : list, optional
        List of categorical feature names. If None, auto-detects based on max_unique_thresh
    encoding_method : str, default='onehot'
        Method to encode categorical variables: 'onehot' or 'label'
    max_unique_thresh : int, default=10
        Maximum number of unique values for a column to be auto-detected as categorical
    drop_original : bool, default=False
        Whether to drop original categorical columns after encoding
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the interaction terms with ID preserved
    list
        List of created interaction feature names
    dict
        Dictionary containing the encoders used for each categorical feature
    """
    # Create a copy of input DataFrame
    df_temp = df.copy()

    # If no features specified, use all columns except the ID column
    if features is None:
        features = [col for col in df_temp.columns if col != exclude_feature]
    elif exclude_feature in features:
        features.remove(exclude_feature)
    
    # Auto-detect categorical features if not specified
    if categorical_features is None:
        categorical_features = [col for col in features 
                              if df_temp[col].nunique() <= max_unique_thresh 
                              and not np.issubdtype(df_temp[col].dtype, np.number)]
    
    numerical_features = [col for col in features if col not in categorical_features]
    
    # Dictionary to store encoders
    encoders = {}
    
    # Handle categorical features
    encoded_features = []
    for cat_feat in categorical_features:
        if encoding_method == 'onehot':
            # Create and fit OneHotEncoder
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded_vals = encoder.fit_transform(df_temp[[cat_feat]])
            
            # Create new column names for encoded features
            encoded_names = [f"{cat_feat}_{val}" for val in encoder.categories_[0]]
            
            # Add encoded columns to dataframe
            for name, column in zip(encoded_names, encoded_vals.T):
                df_temp[name] = column
                encoded_features.append(name)
                
        elif encoding_method == 'label':
            # Create and fit LabelEncoder
            encoder = LabelEncoder()
            df_temp[f"{cat_feat}_encoded"] = encoder.fit_transform(df_temp[cat_feat])
            encoded_features.append(f"{cat_feat}_encoded")
            
        encoders[cat_feat] = encoder
        
        # Drop original categorical column if specified
        if drop_original:
            df_temp = df_temp.drop(columns=[cat_feat])
    
    # Combine numerical and encoded features for interactions
    features_for_interaction = numerical_features + encoded_features
    
    # Generate all possible pairs of features
    feature_pairs = list(combinations(features_for_interaction, 2))
    
    # Create new DataFrame starting with ID column
    interactions_df = pd.DataFrame({exclude_feature: df_temp[exclude_feature]})
    interaction_names = []
    
    # Generate interactions
    for f1, f2 in feature_pairs:
        # Create interaction name
        interaction_name = f"{f1}_{f2}_interaction"
        
        # Compute interaction
        interactions_df[interaction_name] = df_temp[f1] * df_temp[f2]
        interaction_names.append(interaction_name)

    return interactions_df, interaction_names, encoders


def main(test_set_dir, results_dir):
    # Load test set data.
    input_df = pd.read_csv(os.path.join(test_set_dir, "inputs.csv"))

    columns = ['State', 'Sex', 'GeneralHealth', 'AgeCategory',
       'HeightInMeters', 'WeightInKilograms', 'BMI', 'HadAngina', 'HadStroke',
       'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder',
       'HadKidneyDisease', 'HadArthritis', 'HadDiabetes',
       'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
       'DifficultyConcentrating', 'DifficultyWalking',
       'DifficultyDressingBathing', 'DifficultyErrands', 'SmokerStatus',
       'ECigaretteUsage', 'ChestScan', 'RaceEthnicityCategory',
       'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
       'TetanusLast10Tdap', 'HighRiskLastYear', 'CovidPos']

    catagorical_columns = ['State', 'Sex', 'GeneralHealth', 'AgeCategory', 'HadDiabetes',
       'SmokerStatus', 'ECigaretteUsage', 'RaceEthnicityCategory',
       'TetanusLast10Tdap']

    df_feature_enhanced = create_interactions(input_df, features=columns, categorical_features=catagorical_columns, encoding_method='label', 
                       max_unique_thresh=10, drop_original=False)[0].copy()

    # ---------------------------------
    # START PROCESSING TEST SET INPUTS

    # Load the saved model from the file
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model('xgboost_model.json')

    # generate predictions
    output_df = loaded_model.predict_proba(df_feature_enhanced)[:, 1]

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