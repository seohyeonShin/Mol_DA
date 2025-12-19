"""
ChemDFM 스타일 프롬프트 템플릿

Expert chemist 스타일의 상세한 instruction 프롬프트.
각 task에 대해 명확한 지시와 출력 형식을 제공.

Reference: preprocess_for_external.py
"""

CHEMDFM_PROMPTS = {
    # ============ Classification ============
    "bace": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict whether it can inhibit (True) the Beta-site Amyloid Precursor Protein Cleaving Enzyme 1 (BACE1) or cannot inhibit (False). Please answer with only True or False.
SMILES: {smiles}
BACE-1 Inhibit:""",

    "smol-property_prediction-bbbp": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict whether it can penetrate (True) the blood-brain barrier or not (False). Please answer with only True or False.
SMILES: {smiles}
BBB Penetration:""",

    "smol-property_prediction-clintox": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict whether the drug has been approved by FDA (True) or failed clinical trials for toxicity (False). Please answer with only True or False.
SMILES: {smiles}
FDA Approval:""",

    "smol-property_prediction-hiv": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict whether it can inhibit HIV replication (True) or not (False). Please answer with only True or False.
SMILES: {smiles}
HIV Inhibition:""",

    "smol-property_prediction-sider": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict whether it causes hepatobiliary disorders side effect (True) or not (False). Please answer with only True or False.
SMILES: {smiles}
Side Effect:""",

    "tox21": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict whether it shows toxicity (True) or not (False) in the Tox21 assay. Please answer with only True or False.
SMILES: {smiles}
Toxicity:""",

    "toxcast": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict whether it shows toxicity (True) or not (False) in the ToxCast assay. Please answer with only True or False.
SMILES: {smiles}
Toxicity:""",

    # ============ Regression - QM9 ============
    "qm9_homo": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the HOMO (Highest Occupied Molecular Orbital) energy in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
HOMO (Hartree):""",

    "qm9_lumo": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the LUMO (Lowest Unoccupied Molecular Orbital) energy in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
LUMO (Hartree):""",

    "qm9_homo_lumo_gap": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the HOMO-LUMO gap energy in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
Gap (Hartree):""",

    "qm9_dipole_moment": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the dipole moment in Debye units. Answer with only the numerical value.
SMILES: {smiles}
Dipole (Debye):""",

    "qm9_isotropic_polarizability": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the isotropic polarizability in Bohr^3 units. Answer with only the numerical value.
SMILES: {smiles}
Polarizability (Bohr^3):""",

    "qm9_electronic_spatial_extent": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the electronic spatial extent in Bohr^2 units. Answer with only the numerical value.
SMILES: {smiles}
Electronic Spatial Extent (Bohr^2):""",

    "qm9_zero_point_vibrational_energy": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the zero point vibrational energy in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
ZPVE (Hartree):""",

    "qm9_heat_capacity_298K": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the heat capacity at 298.15K in cal/(mol·K) units. Answer with only the numerical value.
SMILES: {smiles}
Cv (cal/mol·K):""",

    "qm9_internal_energy_298K": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the internal energy at 298.15K in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
U_298K (Hartree):""",

    "qm9_enthalpy_298K": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the enthalpy at 298.15K in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
H_298K (Hartree):""",

    "qm9_free_energy_298K": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the free energy at 298.15K in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
G_298K (Hartree):""",

    # ============ Regression - Alchemy ============
    "alchemy_homo": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the HOMO energy in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
HOMO (Hartree):""",

    "alchemy_lumo": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the LUMO energy in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
LUMO (Hartree):""",

    "alchemy_homo_lumo_gap": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the HOMO-LUMO gap energy in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
Gap (Hartree):""",

    # ============ Regression - Solubility/Lipophilicity ============
    "smol-property_prediction-esol": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the aqueous solubility as log solubility (logS in mol/L). Answer with only the numerical value.
SMILES: {smiles}
LogS:""",

    "smol-property_prediction-lipo": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the lipophilicity as logD at pH 7.4. Answer with only the numerical value.
SMILES: {smiles}
LogD:""",

    "aqsol-logS": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the aqueous solubility as logS in mol/L. Answer with only the numerical value.
SMILES: {smiles}
LogS:""",

    "pcqm_homo_lumo_gap": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the HOMO-LUMO gap energy in eV units. Answer with only the numerical value.
SMILES: {smiles}
Gap (eV):""",

    # ============ Reaction Prediction ============
    "forward_reaction_prediction": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the reactants SMILES, predict the major product of the reaction. Answer with only the product SMILES string.
Reactants: {smiles}
Product:""",

    "smol-forward_synthesis": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the reactants SMILES, predict the major product of the reaction. Answer with only the product SMILES string.
Reactants: {smiles}
Product:""",

    "retrosynthesis": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the product SMILES, predict the reactants needed to synthesize it. Answer with only the reactant SMILES (use '.' to separate multiple reactants).
Product: {smiles}
Reactants:""",

    "smol-retrosynthesis": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the product SMILES, predict the reactants needed to synthesize it. Answer with only the reactant SMILES (use '.' to separate multiple reactants).
Product: {smiles}
Reactants:""",

    "reagent_prediction": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the reactants and product of a reaction, predict the reagents or catalysts needed. Answer with only the reagent SMILES (use '.' to separate multiple reagents).
Reaction: {smiles}
Reagents:""",

    # Presto variants
    "presto-forward_reaction_prediction": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the reactants SMILES, predict the major product of the reaction. Answer with only the product SMILES string.
Reactants: {smiles}
Product:""",

    "presto-retrosynthesis": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the product SMILES, predict the reactants needed to synthesize it. Answer with only the reactant SMILES (use '.' to separate multiple reactants).
Product: {smiles}
Reactants:""",

    "presto-reagent_prediction": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the reactants and product of a reaction, predict the reagents or catalysts needed. Answer with only the reagent SMILES.
Reaction: {smiles}
Reagents:""",

    # Orderly variants
    "orderly-forward_reaction_prediction": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the reactants SMILES, predict the major product of the reaction. Answer with only the product SMILES string.
Reactants: {smiles}
Product:""",

    "orderly-retrosynthesis": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the product SMILES, predict the reactants needed to synthesize it. Answer with only the reactant SMILES (use '.' to separate multiple reactants).
Product: {smiles}
Reactants:""",

    "orderly-reagent_prediction": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the reactants and product of a reaction, predict the reagents or catalysts needed. Answer with only the reagent SMILES.
Reaction: {smiles}
Reagents:""",

    # ============ Molecule Generation (Text2Mol) ============
    "chebi-20-text2mol": """You are an expert chemist, your task is to generate molecules from descriptions.
Please strictly follow the format, no other information can be provided. Given the description of a molecule, generate the corresponding SMILES string. Answer with only the SMILES string.
Description: {description}
SMILES:""",

    "smol-molecule_generation": """You are an expert chemist, your task is to generate molecules from descriptions.
Please strictly follow the format, no other information can be provided. Given the description of a molecule, generate the corresponding SMILES string. Answer with only the SMILES string.
Description: {description}
SMILES:""",

    # ============ Molecule Captioning (Mol2Text) ============
    "chebi-20-mol2text": """You are an expert chemist, your task is to describe molecules.
Given the SMILES string of a molecule, provide a detailed description of its structure, functional groups, and properties.
SMILES: {smiles}
Description:""",

    "smol-molecule_captioning": """You are an expert chemist, your task is to describe molecules.
Given the SMILES string of a molecule, provide a detailed description of its structure, functional groups, and properties.
SMILES: {smiles}
Description:""",
}
