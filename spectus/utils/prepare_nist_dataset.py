"""
Standalone reproduction of notebooks/2_data_preprocessing.ipynb, section 1 (NIST dataset).

Loads data/nist/mainlib.msp, drops spectra with no usable identifier, reconstructs
missing SMILES from InChI where possible, canonicalizes + destereochemicalizes every
SMILES, splits into train/valid/test (80/10/10, grouped by unique SMILES so a molecule's
spectra never leak across splits), and writes {train,valid,test}.{msp,jsonl,smi} to
data/nist/. Does not touch the ZINC/RASSP/NEIMS synthetic-data sections of the notebook.
"""

from pathlib import Path
import random

import numpy as np
from rdkit import Chem
from tqdm import tqdm

import matchms
from matchms.importing import load_from_msp
from matchms.exporting import save_as_msp

try:
    from .spectra_process_utils import remove_stereochemistry_and_canonicalize, msp2jsonl, jsonl2smi
except ImportError:
    from utils.spectra_process_utils import remove_stereochemistry_and_canonicalize, msp2jsonl, jsonl2smi

SEED = 42
TRAIN_RATIO, VALID_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
TRAIN_INDEX, VALID_INDEX, TEST_INDEX = 0, 1, 2


def reconstruct_information(dataset):
    updated_dataset = []
    for spectrum in tqdm(dataset, desc="reconstructing identifiers"):
        smiles = spectrum.get("smiles")
        if smiles is None and spectrum.get("inchikey") is None and spectrum.get("inchi") is None:
            continue

        if (smiles is None
                or Chem.MolFromSmiles(smiles, sanitize=False) is None
                or Chem.MolFromSmiles(smiles) is None) and spectrum.get("inchi") is not None:
            tmp = Chem.inchi.MolFromInchi(spectrum.get("inchi"))
            if tmp is not None:
                spectrum.set("smiles", Chem.MolToSmiles(tmp))
                smiles = spectrum.get("smiles")

        if smiles is None and spectrum.get("inchi") is not None:
            spectrum = matchms.filtering.derive_smiles_from_inchi(spectrum)
            smiles = spectrum.get("smiles")

        if smiles is None:
            continue

        updated_dataset.append(spectrum)
    return updated_dataset


def canonicalize_dataset(dataset):
    updated_dataset = []
    changed = 0
    for spectrum in tqdm(dataset, desc="canonicalizing SMILES"):
        smiles = spectrum.get("smiles")
        if smiles is None:
            raise ValueError("Smiles is None, reconstruction and filtering poorly done.")
        new_smiles = remove_stereochemistry_and_canonicalize(smiles)
        if new_smiles is None:
            continue
        spectrum.set("smiles", new_smiles)
        if new_smiles != smiles:
            changed += 1
        updated_dataset.append(spectrum)
    print(f"Number of smiles canonicalized or destereochemicalized: {changed}")
    return updated_dataset


def unique_mapping(dataset):
    smiles_dict = {}
    counter_none = 0
    for spectrum in dataset:
        smiles = spectrum.get("smiles")
        if smiles is None:
            counter_none += 1
            continue
        smiles_dict.setdefault(smiles, []).append(spectrum)
    print(f"Missing smiles identifier in {counter_none} cases")
    return smiles_dict


def generate_index(dataset, train_ratio, valid_ratio, test_ratio):
    n = len(dataset)
    train_idx = np.full(int(n * train_ratio), TRAIN_INDEX, dtype=int)
    valid_idx = np.full(int(n * valid_ratio), VALID_INDEX, dtype=int)
    test_idx = np.full(n - len(train_idx) - len(valid_idx), TEST_INDEX, dtype=int)
    idx = np.concatenate((train_idx, valid_idx, test_idx))
    np.random.shuffle(idx)
    return idx


def generate_train_test_dataset(dataset, indices):
    train, valid, test = [], [], []
    for i, smiles in zip(indices, dataset):
        target = {TRAIN_INDEX: train, VALID_INDEX: valid, TEST_INDEX: test}[i]
        target.extend(dataset[smiles])
    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)
    return train, valid, test


def save_dataset(dataset, path: Path, name: str):
    path.mkdir(mode=0o777, parents=True, exist_ok=True)
    save_as_msp(dataset, str(path / f"{name}.msp"))


def main():
    repo_root = Path(__file__).resolve().parents[2]
    nist_dir = repo_root / "data" / "nist"
    nist_path = nist_dir / "mainlib.msp"

    np.random.seed(SEED)
    random.seed(SEED)

    print(f"Loading {nist_path}")
    nist_dataset = list(load_from_msp(str(nist_path), metadata_harmonization=False))
    print(f"Loaded {len(nist_dataset)} spectra")

    reconstructed = reconstruct_information(nist_dataset)
    print(f"{len(reconstructed)} / {len(nist_dataset)} spectra remain after identifier filtering/reconstruction")

    canonicalized = canonicalize_dataset(reconstructed)
    unique_smiles = {s.get("smiles") for s in canonicalized}
    print(f"{len(canonicalized)} / {len(reconstructed)} spectra remain after canonicalization; "
          f"{len(unique_smiles)} unique SMILES")

    nist_dict = unique_mapping(canonicalized)
    indices = generate_index(nist_dict, TRAIN_RATIO, VALID_RATIO, TEST_RATIO)
    train, valid, test = generate_train_test_dataset(nist_dict, indices)

    train_smi = {s.get("smiles") for s in train}
    valid_smi = {s.get("smiles") for s in valid}
    test_smi = {s.get("smiles") for s in test}
    print(f"train={len(train)} valid={len(valid)} test={len(test)}")
    print(f"overlap train/valid={len(train_smi & valid_smi)} "
          f"train/test={len(train_smi & test_smi)} valid/test={len(valid_smi & test_smi)}")

    save_dataset(train, nist_dir, "train")
    save_dataset(valid, nist_dir, "valid")
    save_dataset(test, nist_dir, "test")

    for dataset_type in ["train", "valid", "test"]:
        msp2jsonl(
            path_msp=nist_dir / f"{dataset_type}.msp",
            path_jsonl=nist_dir / f"{dataset_type}.jsonl",
            tokenizer=None,
            keep_spectra=True,
            do_preprocess=False,
        )
        jsonl2smi(str(nist_dir / f"{dataset_type}.jsonl"))

    print("Done.")


if __name__ == "__main__":
    main()
