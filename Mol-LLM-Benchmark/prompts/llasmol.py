"""
LlaSMol 스타일 프롬프트 템플릿 (SMolInstruct 원본 스타일)

Uses <SMILES>...</SMILES> tags for canonicalization
Response uses tags: <NUMBER>, <BOOLEAN>, <SMILES>, <MOLFORMULA>

Reference:
- osunlp/SMolInstruct (https://huggingface.co/datasets/osunlp/SMolInstruct)
- LlaSMol GitHub: https://github.com/OSU-NLP-Group/LLM4Chem

SMolInstruct uses natural question format, NOT instruction format.
"""

LLASMOL_PROMPTS = {
    # ============ Classification ============
    # SMolInstruct style: natural question format
    "bace": """Can <SMILES> {smiles} </SMILES> inhibit human β-secretase 1 (BACE-1)?""",

    "smol-property_prediction-bbbp": """Is blood-brain barrier permeability (BBBP) a property of <SMILES> {smiles} </SMILES> ?""",

    "smol-property_prediction-clintox": """Is <SMILES> {smiles} </SMILES> toxic?""",

    "smol-property_prediction-hiv": """Can <SMILES> {smiles} </SMILES> serve as an inhibitor of HIV replication?""",

    "smol-property_prediction-sider": """Are there any known side effects of <SMILES> {smiles} </SMILES> affecting the heart?""",

    "tox21": """Is <SMILES> {smiles} </SMILES> toxic in the Tox21 assay?""",

    "toxcast": """Is <SMILES> {smiles} </SMILES> toxic in the ToxCast assay?""",

    # ============ Regression - Solubility/Lipophilicity ============
    # SMolInstruct example: "How soluble is <SMILES> CC(C)Cl </SMILES> ?"
    "smol-property_prediction-esol": """How soluble is <SMILES> {smiles} </SMILES> ?""",

    "smol-property_prediction-lipo": """Predict the octanol/water distribution coefficient logD under the circumstance of pH 7.4 for <SMILES> {smiles} </SMILES> .""",

    "aqsol-logS": """How soluble is <SMILES> {smiles} </SMILES> ?""",

    # ============ Regression - QM9 ============
    "qm9_homo": """What is the HOMO energy of <SMILES> {smiles} </SMILES> in Hartree?""",

    "qm9_lumo": """What is the LUMO energy of <SMILES> {smiles} </SMILES> in Hartree?""",

    "qm9_homo_lumo_gap": """What is the HOMO-LUMO gap of <SMILES> {smiles} </SMILES> in Hartree?""",

    "qm9_dipole_moment": """What is the dipole moment of <SMILES> {smiles} </SMILES> in Debye?""",

    "qm9_isotropic_polarizability": """What is the isotropic polarizability of <SMILES> {smiles} </SMILES> in Bohr³?""",

    "qm9_electronic_spatial_extent": """What is the electronic spatial extent of <SMILES> {smiles} </SMILES> in Bohr²?""",

    "qm9_zero_point_vibrational_energy": """What is the zero point vibrational energy of <SMILES> {smiles} </SMILES> in Hartree?""",

    "qm9_heat_capacity_298K": """What is the heat capacity at 298.15K of <SMILES> {smiles} </SMILES> in cal/(mol·K)?""",

    "qm9_internal_energy_298K": """What is the internal energy at 298.15K of <SMILES> {smiles} </SMILES> in Hartree?""",

    "qm9_enthalpy_298K": """What is the enthalpy at 298.15K of <SMILES> {smiles} </SMILES> in Hartree?""",

    "qm9_free_energy_298K": """What is the free energy at 298.15K of <SMILES> {smiles} </SMILES> in Hartree?""",

    # ============ Regression - Alchemy ============
    "alchemy_homo": """What is the HOMO energy of <SMILES> {smiles} </SMILES> in Hartree?""",

    "alchemy_lumo": """What is the LUMO energy of <SMILES> {smiles} </SMILES> in Hartree?""",

    "alchemy_homo_lumo_gap": """What is the HOMO-LUMO gap of <SMILES> {smiles} </SMILES> in Hartree?""",

    "pcqm_homo_lumo_gap": """What is the HOMO-LUMO gap of <SMILES> {smiles} </SMILES> in eV?""",

    # ============ Mol2Text ============
    # SMolInstruct example: "Describe this molecule: <SMILES> ... </SMILES>"
    "chebi-20-mol2text": """Describe this molecule: <SMILES> {smiles} </SMILES>""",

    "smol-molecule_captioning": """Describe this molecule: <SMILES> {smiles} </SMILES>""",

    # ============ Text2Mol ============
    # SMolInstruct example: "Give me a molecule that satisfies the conditions outlined in the description: ..."
    "chebi-20-text2mol": """Give me a molecule that satisfies the conditions outlined in the description: {description}""",

    "smol-molecule_generation": """Give me a molecule that satisfies the conditions outlined in the description: {description}""",

    # ============ Reaction - Forward Synthesis ============
    # SMolInstruct example: "<SMILES> ... </SMILES> Based on the reactants and reagents given above, suggest a possible product."
    "forward_reaction_prediction": """<SMILES> {smiles} </SMILES> Based on the reactants and reagents given above, suggest a possible product.""",

    "smol-forward_synthesis": """<SMILES> {smiles} </SMILES> Based on the reactants and reagents given above, suggest a possible product.""",

    "presto-forward_reaction_prediction": """<SMILES> {smiles} </SMILES> Based on the reactants and reagents given above, suggest a possible product.""",

    "orderly-forward_reaction_prediction": """<SMILES> {smiles} </SMILES> Based on the reactants and reagents given above, suggest a possible product.""",

    # ============ Reaction - Retrosynthesis ============
    # SMolInstruct example: "Identify possible reactants that could have been used to create the specified product. <SMILES> ... </SMILES>"
    "retrosynthesis": """Identify possible reactants that could have been used to create the specified product. <SMILES> {smiles} </SMILES>""",

    "smol-retrosynthesis": """Identify possible reactants that could have been used to create the specified product. <SMILES> {smiles} </SMILES>""",

    "presto-retrosynthesis": """Identify possible reactants that could have been used to create the specified product. <SMILES> {smiles} </SMILES>""",

    "orderly-retrosynthesis": """Identify possible reactants that could have been used to create the specified product. <SMILES> {smiles} </SMILES>""",

    # ============ Reaction - Reagent Prediction ============
    "reagent_prediction": """Given the reactants and product of a reaction <SMILES> {smiles} </SMILES>, predict the reagents or catalysts needed.""",

    "presto-reagent_prediction": """Given the reactants and product of a reaction <SMILES> {smiles} </SMILES>, predict the reagents or catalysts needed.""",

    "orderly-reagent_prediction": """Given the reactants and product of a reaction <SMILES> {smiles} </SMILES>, predict the reagents or catalysts needed.""",
}
