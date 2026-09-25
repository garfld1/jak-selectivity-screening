"""
Screen a new compound library using a pretrained RF Morgan fingerprint model.

Input:
    A CSV/TSV containing a "smiles" column.

Model:
    results/qsar/rf_final_model.joblib

Output:
    A CSV containing all valid compounds with predicted delta_pIC50,
    plus a selected top fraction or top-N subset.

Examples:
    python src/qsar_screen.py \
        --input coconut.csv \
        --top 5pct

    python src/qsar_screen.py \
        --input coconut.csv \
        --top 4000
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


MODEL_PATH = "results/qsar/rf_final_model.joblib"


def generate_morgan_fingerprints(
    smiles_list,
    radius,
    n_bits,
    include_chirality,
    use_bond_types,
):
    """Generate Morgan fingerprints using the same settings as training."""

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=include_chirality,
        useBondTypes=use_bond_types,
    )

    fps = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            continue

        fp = fp_gen.GetFingerprintAsNumPy(mol)

        fps.append(fp)
        valid_indices.append(i)

    if not fps:
        raise ValueError("No valid SMILES found.")

    return (
        np.asarray(fps, dtype=np.float32),
        np.asarray(valid_indices, dtype=int),
    )


def get_top_n(df, top_choice):
    """Return the requested top-ranked compounds."""

    n = len(df)

    if top_choice == "1pct":
        k = max(1, int(np.ceil(n * 0.01)))

    elif top_choice == "5pct":
        k = max(1, int(np.ceil(n * 0.05)))

    elif top_choice == "10pct":
        k = max(1, int(np.ceil(n * 0.10)))

    else:
        # Hardcoded number, e.g. --top 4000
        k = int(top_choice)

        if k < 1:
            raise ValueError("Top-N must be at least 1.")

    k = min(k, n)

    return df.head(k).copy()


def main():
    parser = argparse.ArgumentParser(
        description="Screen a compound library with a pretrained QSAR model."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to screening CSV/TSV containing a 'smiles' column.",
    )

    parser.add_argument(
        "--top",
        required=True,
        help="Selection: 1pct, 5pct, 10pct, or a hardcoded number such as 4000.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/qsar_screen",
        help="Directory for screening results.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load screening dataframe
    # ------------------------------------------------------------

    if args.input.endswith((".tsv", ".txt")):
        df = pd.read_csv(args.input, sep="\t")
    else:
        df = pd.read_csv(args.input)

    if "smiles" not in df.columns:
        raise ValueError("Input dataframe must contain a 'smiles' column.")

    print(f"Loaded {len(df):,} compounds.")

    # ------------------------------------------------------------
    # Load pretrained model
    # ------------------------------------------------------------

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    fp_settings = artifact["fingerprint"]

    print("Loaded pretrained model.")
    print(f"Model: {MODEL_PATH}")
    print(
        f"Morgan radius: {fp_settings['radius']}, "
        f"bits: {fp_settings['n_bits']}"
    )

    # ------------------------------------------------------------
    # Generate fingerprints
    # ------------------------------------------------------------

    X, valid_indices = generate_morgan_fingerprints(
        df["smiles"].astype(str).tolist(),
        radius=fp_settings["radius"],
        n_bits=fp_settings["n_bits"],
        include_chirality=fp_settings["include_chirality"],
        use_bond_types=fp_settings["use_bond_types"],
    )

    screening_df = df.iloc[valid_indices].copy()

    print(f"Valid SMILES: {len(screening_df):,}")
    print(
        f"Invalid SMILES removed: "
        f"{len(df) - len(screening_df):,}"
    )

    # ------------------------------------------------------------
    # Predict delta_pIC50
    # ------------------------------------------------------------

    predictions = model.predict(X)

    screening_df["predicted_delta_pIC50"] = predictions

    # ------------------------------------------------------------
    # Rank compounds
    # ------------------------------------------------------------

    screening_df = screening_df.sort_values(
        "predicted_delta_pIC50",
        ascending=False,
    ).reset_index(drop=True)

    screening_df["QSAR_rank"] = np.arange(
        1,
        len(screening_df) + 1,
    )

    # ------------------------------------------------------------
    # Select requested top compounds
    # ------------------------------------------------------------

    top_df = get_top_n(
        screening_df,
        args.top,
    )

    print()
    print(f"Selected {len(top_df):,} compounds.")

    print(
        f"Predicted delta_pIC50 range: "
        f"{top_df['predicted_delta_pIC50'].min():.3f} "
        f"to "
        f"{top_df['predicted_delta_pIC50'].max():.3f}"
    )

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    all_output = os.path.join(
        args.output_dir,
        "all_predictions.csv",
    )

    top_output = os.path.join(
        args.output_dir,
        f"top_{args.top}.csv",
    )

    screening_df.to_csv(
        all_output,
        index=False,
    )

    top_df.to_csv(
        top_output,
        index=False,
    )

    print()
    print("Saved:")
    print(f"  All predictions: {all_output}")
    print(f"  Selected compounds: {top_output}")


if __name__ == "__main__":
    main()