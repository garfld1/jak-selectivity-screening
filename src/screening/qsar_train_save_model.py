import argparse
import os

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor


RADIUS = 2
N_BITS = 2048
RANDOM_STATE = 42


def generate_morgan_fingerprints(smiles_list):

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=N_BITS,
        includeChirality=True,
        useBondTypes=True,
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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to qsar_df CSV/TSV",
    )

    parser.add_argument(
        "--output",
        default="results/qsar/rf_final_model.joblib",
        help="Path for saved model",
    )

    args = parser.parse_args()

    # Load data
    if args.input.endswith((".tsv", ".txt")):
        df = pd.read_csv(
            args.input,
            sep="\t",
        )
    else:
        df = pd.read_csv(args.input)

    # Check required columns
    required = {"smiles", "delta_pIC50"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    print(
        f"Training on {len(df)} compounds..."
    )

    # Generate fingerprints
    X, valid_indices = generate_morgan_fingerprints(
        df["smiles"].astype(str).tolist()
    )

    # Targets
    y = df["delta_pIC50"].to_numpy(
        dtype=float
    )[valid_indices]

    print(
        f"Using {len(y)} valid compounds."
    )

    # Train final model on ALL data
    model = RandomForestRegressor(
        random_state=RANDOM_STATE
    )

    model.fit(X, y)

    # Save model + fingerprint settings
    artifact = {
        "model": model,
        "fingerprint": {
            "type": "morgan",
            "radius": RADIUS,
            "n_bits": N_BITS,
            "include_chirality": True,
            "use_bond_types": True,
        },
    }

    os.makedirs(
        os.path.dirname(args.output),
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        args.output,
    )

    print(
        f"Saved pretrained model to:"
    )
    print(args.output)


if __name__ == "__main__":
    main()