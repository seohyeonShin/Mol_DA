"""
기본 프롬프트 템플릿

특정 task에 매핑되지 않은 경우 task type별로 사용하는 fallback 프롬프트.
"""

DEFAULT_PROMPTS = {
    "classification": """You are an expert chemist, your task is to predict the property of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the property. Please answer with only True or False.
SMILES: {smiles}
Answer:""",

    "regression": """You are an expert chemist, your task is to predict the property of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the numerical property value. Answer with only the numerical value.
SMILES: {smiles}
Value:""",

    "reaction": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Answer with only the SMILES string.
Input: {smiles}
Output:""",

    "text2mol": """You are an expert chemist, your task is to generate molecules from descriptions.
Please strictly follow the format, no other information can be provided. Answer with only the SMILES string.
Description: {description}
SMILES:""",

    "mol2text": """You are an expert chemist, your task is to describe molecules.
Given the SMILES string of a molecule, provide a description.
SMILES: {smiles}
Description:""",
}
