"""
Deterministic molecular featurization utilities.

Produces a fixed-order feature vector:
    [ECFP bits..., RDKit descriptors in DESCRIPTOR_ORDER]

Use the same Morgan radius/bit-length you used in training.

Example:
    from ml_featurizer import featurize_smiles, DESCRIPTOR_ORDER
    x = featurize_smiles("CCO", radius=2, n_bits=2048)
"""

from typing import List, Dict
import numpy as np

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski
from rdkit import DataStructs

# Fixed descriptor order (must match training if training used these)
DESCRIPTOR_ORDER: List[str] = [
    # Lipinski counts
    "HBD", "HBA", "RotB",
    # Structural
    "RingCount", "AromaticRingCount", "FractionCSP3",
    "HeavyAtomCount", "HeteroAtomCount",
    # Substructure flags
    "HasPyridine", "HasPyrimidine",
    # Simple normalizations (Core physicochemical)
    "MW_norm", "TPSA_norm", "LogP_norm", "SP3_norm",
]

# SMARTS used for substructure flags
_PYRIDINE   = Chem.MolFromSmarts("n1ccccc1")
_PYRIMIDINE = Chem.MolFromSmarts("n1cnccc1")


def morgan_bits_from_mol(mol: Chem.Mol, radius: int, n_bits: int) -> np.ndarray:
    """Return ECFP bit vector as float32 of shape (n_bits,)."""
    bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)


def compute_rdkit_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """Compute the descriptor dict keyed by DESCRIPTOR_ORDER names."""
    # Physicochemical
    mol_wt = Descriptors.MolWt(mol)
    logp   = Descriptors.MolLogP(mol)
    tpsa   = Descriptors.TPSA(mol)

    # Lipinski
    num_hbd = Lipinski.NumHDonors(mol)
    num_hba = Lipinski.NumHAcceptors(mol)
    num_rotb = Lipinski.NumRotatableBonds(mol)

    # Structural
    ring_count          = rdMolDescriptors.CalcNumRings(mol)
    aromatic_ring_count = rdMolDescriptors.CalcNumAromaticRings(mol)
    fraction_sp3        = rdMolDescriptors.CalcFractionCSP3(mol)
    heavy_atom_count    = Descriptors.HeavyAtomCount(mol)
    heteroatom_count    = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))

    # Substructures
    has_pyridine   = int(mol.HasSubstructMatch(_PYRIDINE))
    has_pyrimidine = int(mol.HasSubstructMatch(_PYRIMIDINE))

    # Simple normalizations (keep consistent with training)
    mw_norm   = mol_wt / 900.0
    tpsa_norm = tpsa / 300.0
    logp_norm = (logp + 5.0) / 10.0
    sp3_norm  = fraction_sp3

    return {
        "HBD": num_hbd,
        "HBA": num_hba,
        "RotB": num_rotb,
        "RingCount": ring_count,
        "AromaticRingCount": aromatic_ring_count,
        "FractionCSP3": fraction_sp3,
        "HeavyAtomCount": heavy_atom_count,
        "HeteroAtomCount": heteroatom_count,
        "HasPyridine": has_pyridine,
        "HasPyrimidine": has_pyrimidine,
        "MW_norm": mw_norm,
        "TPSA_norm": tpsa_norm,
        "LogP_norm": logp_norm,
        "SP3_norm": sp3_norm,
    }


def featurize_smiles(smiles: str, radius: int, n_bits: int) -> np.ndarray:
    """
    SMILES - concatenated features of length (n_bits + len(DESCRIPTOR_ORDER)) as float32.
    On parse failure, returns zeros of the correct length.
    """
    if not isinstance(smiles, str):
        smiles = ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits + len(DESCRIPTOR_ORDER),), dtype=np.float32)

    bits = morgan_bits_from_mol(mol, radius, n_bits)
    desc = compute_rdkit_descriptors(mol)
    desc_vec = np.array([float(desc.get(k, 0.0)) for k in DESCRIPTOR_ORDER], dtype=np.float32)

    return np.concatenate([desc_vec, bits]).astype(np.float32)
