import argparse
import pandas as pd
import numpy as np
from src.pipeline import run_pipeline
from src.config import CONFIG
import os

def main():
    parser = argparse.ArgumentParser(description="Run inhibitor classification pipeline.")
    parser.add_argument("train_csv", help="Path to training CSV file.")
    parser.add_argument("test_csv", help="Path to test CSV file.")
    parser.add_argument("--selected_models", nargs="+", default=None,
                        help="List of model keys to train (default: all).")
    parser.add_argument("--voting", choices=["hard", "soft"], default="hard",
                        help="Voting type for ensemble.")
    args = parser.parse_args()

    # Load data
    df_train = pd.read_csv(args.train_csv)
    df_test = pd.read_csv(args.test_csv)

    # Your preprocessing here - select numeric features, drop unwanted cols etc.
    non_features = {"y", "class", "MurckoScaffold", "standard_value" ,"canonical_smiles", "standardized_smiles", "molecule_chembl_id"}
    train_feat_cols = [c for c in df_train.columns if c not in non_features and pd.api.types.is_numeric_dtype(df_train[c])]
    test_feat_cols = [c for c in df_test.columns if c not in non_features and pd.api.types.is_numeric_dtype(df_test[c])]
    common_cols = [c for c in train_feat_cols if c in test_feat_cols]

    X_train = df_train[common_cols].values
    y_train = df_train["y"].astype(int).values
    X_test = df_test[common_cols].values
    y_test = df_test["y"].astype(int).values

    # Ensure output dir exists
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Run pipeline
    summary = run_pipeline(X_train, y_train, X_test, y_test,
                           selected_models=args.selected_models,
                           voting=args.voting)

    print(f"Pipeline complete. Models and metrics saved to {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()

