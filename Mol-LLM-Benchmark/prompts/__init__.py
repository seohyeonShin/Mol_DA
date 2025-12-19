"""
Prompt Templates for Mol-LLM-Benchmark

프롬프트 템플릿 모듈.
- chemdfm: ChemDFM 스타일 (expert chemist instruction format)
- llasmol: LlaSMol/SMolInstruct 스타일 (natural question format with <SMILES> tags)
- defaults: 기본 fallback 프롬프트 (task type별)
"""

from .chemdfm import CHEMDFM_PROMPTS
from .llasmol import LLASMOL_PROMPTS
from .defaults import DEFAULT_PROMPTS

__all__ = [
    "CHEMDFM_PROMPTS",
    "LLASMOL_PROMPTS",
    "DEFAULT_PROMPTS",
]
