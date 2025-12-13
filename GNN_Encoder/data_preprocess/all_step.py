#!/usr/bin/env python3
"""
Mol-LLM Stage 1: Data Preparation Pipeline v3
==============================================

논문 기준 (arXiv:2502.02810v2, Appendix B.1, B.2):
- Step 1: Compute Functional Groups (85개, RDKit Fragments 모듈)
- Step 2: Sparsity-aware Importance Sampling (논문 수식 정확히 구현)
         → 11 most prevalent + 1 rarest 제외 → K=72개 FG 사용
         → 5M molecules 샘플링
- Step 3: SELFIES Conversion + Vocabulary 구축
- Step 4: PyTorch Geometric Graph 변환

출력 파일:
- pubchem_with_fg.csv: 전체 PubChem + 85 FGs
- pubchem_5M_sampled.csv: 5M 샘플 + 85 FGs
- pubchem_5M_sampled_filtered.csv: 5M 샘플 + 72 FGs (training용)
- selfies_vocab.json: SELFIES vocabulary
- graphs_5M.pt: PyTorch Geometric Data 리스트
- fg_distribution_figure6.png: 논문 Figure 6 재현
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from torch_geometric.data import InMemoryDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import selfies as sf
import torch
from ogb.utils import smiles2graph
from rdkit import Chem, RDLogger
from rdkit.Chem import Fragments
from torch_geometric.data import Data
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')


# ============================================
# Configuration (Dataclass)
# ============================================
@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # 경로
    data_dir: Path = field(default_factory=lambda: Path("/workspace/Origin/mol_llm/Mol-LLM/data"))
    
    # 샘플링 설정 (논문 B.1)
    n_samples: int = 5_300_000  # 논문: "sample 5M molecules"
    random_seed: int = 42
    max_seq_len: int = 512  # 논문 B.2: "sequences are truncated to max length of 512"
    n_cores: int = 64
    
    # 제외할 FG 개수 (논문 B.1: "12 most prevalent + 1 rarest")
    n_most_prevalent: int = 12
    n_rarest: int = 1
    
    # 85개 전체 FG (RDKit Fragments 모듈)
    # 논문: "87 functional groups" - RDKit 버전에 따라 85개 사용 가능
    rdkit_props_all: List[str] = field(default_factory=lambda: [
        'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO',
        'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O',
        'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1',
        'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
        'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate',
        'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine',
        'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur',
        'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
        'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan',
        'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole',
        'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
        'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',
        'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom',
        'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime',
        'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond',
        'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine',
        'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
        'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene',
        'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene',
        'fr_unbrch_alkane', 'fr_urea'
    ])
    @property
    def output_graphs_chunks_dir(self) -> Path:
        return self.data_dir / "graphs_5M_chunks" # <--- 청크 저장 경로 추가
    
    #step1 input
    @property
    def input_smiles(self) -> Path:
        return self.data_dir / "CID-SMILES"
    
    #step1 output, step2 input
    @property
    def output_fg_csv(self) -> Path:
        return self.data_dir / "pubchem_with_fg.csv"
    
    #step2 output
    @property
    def output_sampled_csv(self) -> Path:
        return self.data_dir / "pubchem_5M_sampled.csv"
    
    #step2 output, step3 input
    @property
    def output_filtered_csv(self) -> Path:
        return self.data_dir / "pubchem_5M_sampled_filtered.csv"
    
    #step3 output
    @property
    def output_selfies_csv(self) -> Path:
        return self.data_dir / "pubchem_5M_with_selfies.csv"
    #step3 output, step4 input
    @property
    def output_vocab_json(self) -> Path:
        return self.data_dir / "selfies_vocab.json"
    
    @property
    def output_graphs_train_pt(self) -> Path:
        return self.data_dir / "graphs_5M_train_final.pt"
    @property
    def output_graphs_pt(self) -> Path:
        return self.data_dir / "graphs_5M.pt"
    
    @property
    def output_figure(self) -> Path:
        return self.data_dir / "fg_distribution_figure6.png"


# Global config
CONFIG = PipelineConfig()






class _CollatedDataset(InMemoryDataset):
    """
    (data, slices) 튜플로부터 개별 Data를 복원하기 위한 간단한 래퍼.
    step4_graph_conversion 내부에서만 사용.
    """
    def __init__(self, data, slices, max_seq_len: int):
        super().__init__()
        self.data = data
        self.slices = slices
        self.max_seq_len = max_seq_len

    def __len__(self):
        # slices['x']의 길이 - 1 이 샘플 개수
        return len(self.slices['x']) - 1

    def get(self, idx):
        data = super().get(idx)

        # 필요하면 SELFIES 길이 잘라주기 (훈련 코드와 일관성 유지)
        if hasattr(data, "selfies_tokens"):
            if isinstance(data.selfies_tokens, list):
                data.selfies_tokens = torch.tensor(data.selfies_tokens, dtype=torch.long)
            if data.selfies_tokens.size(0) > self.max_seq_len:
                data.selfies_tokens = data.selfies_tokens[:self.max_seq_len]

        return data


# ============================================
# Step 1: Compute Functional Groups
# ============================================
def _compute_fg_single(smiles: str) -> Optional[Tuple[str, List[int]]]:
    """단일 SMILES → (smiles, fg_vector) or None"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        fg_vector = []
        for fg_name in CONFIG.rdkit_props_all:
            func = getattr(Fragments, fg_name)
            fg_vector.append(func(mol))
        
        return (smiles, fg_vector)
    except Exception:
        return None


def step1_compute_fg(config: PipelineConfig = CONFIG) -> pd.DataFrame:
    """
    Step 1: 전체 PubChem에 대해 85개 FG 계산
    
    논문 B.1: "Leveraging the RDKit Fragments module, we enumerate 87 functional groups"
    (RDKit 버전에 따라 85개 사용 가능)
    """
    print("=" * 70)
    print("Step 1: Computing Functional Groups")
    print("=" * 70)
    
    rdkit_props = config.rdkit_props_all
    print(f"Total FG count: {len(rdkit_props)}")
    
    # CID-SMILES 로드
    print(f"\nLoading {config.input_smiles}...")
    df = pd.read_csv(config.input_smiles, sep='\t', names=['CID', 'SMILES'])
    print(f"Total molecules: {len(df):,}")
    
    smiles_list = df['SMILES'].tolist()
    
    # 멀티프로세싱
    print(f"\nComputing FGs using {config.n_cores} cores...")
    with Pool(config.n_cores) as pool:
        results = list(tqdm(
            pool.imap(_compute_fg_single, smiles_list, chunksize=5000),
            total=len(smiles_list),
            desc="Computing FGs"
        ))
    
    # 유효한 결과만 필터링
    valid_results = [r for r in results if r is not None]
    print(f"Valid molecules: {len(valid_results):,} / {len(results):,} "
          f"({100 * len(valid_results) / len(results):.2f}%)")
    
    # DataFrame 생성
    smiles_valid = [r[0] for r in valid_results]
    fg_data = [r[1] for r in valid_results]
    
    result_df = pd.DataFrame({'SMILES': smiles_valid})
    fg_df = pd.DataFrame(fg_data, columns=rdkit_props)
    result_df = pd.concat([result_df, fg_df], axis=1)
    
    # 저장
    print(f"\nSaving to {config.output_fg_csv}...")
    result_df.to_csv(config.output_fg_csv, index=False)
    print(f"Shape: {result_df.shape} (SMILES + {len(rdkit_props)} FGs)")
    
    return result_df


# ============================================
# Step 2: Sparsity-aware Importance Sampling
# ============================================
def step2_sparsity_sampling(
    df: Optional[pd.DataFrame] = None,
    config: PipelineConfig = CONFIG
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Step 2: Sparsity-aware Importance Sampling + Figure 6 시각화
    
    논문 B.1 수식:
    - c_g = Σ_i x_{i,g} : group g의 빈도
    - s_g = 1 / (c_g + ε) : sparsity factor
    - σ_i = (Σ_g x_{i,g} · s_g)² : molecule i의 sparsity score
    - p_i = σ_i / Σ_j σ_j : sampling probability
    
    제외 기준 (논문 B.1):
    - "11 most prevalent groups (from fr_NH0 to fr_aryl_methyl)"
    - "the rarest group (fr_prisulfonamd)"
    - "retaining K = 72 functional groups"
    
    Returns:
        sampled_df: 샘플링된 DataFrame
        excluded_fgs: 제외된 FG 리스트
        retained_fgs: 사용할 72개 FG 리스트
    """
    print("\n" + "=" * 70)
    print("Step 2: Sparsity-aware Importance Sampling")
    print("=" * 70)
    
    if df is None:
        print(f"Loading {config.output_fg_csv}...")
        df = pd.read_csv(config.output_fg_csv)
    
    rdkit_props_all = config.rdkit_props_all
    print(f"Total molecules: {len(df):,}")
    print(f"Total FGs: {len(rdkit_props_all)}")
    
    # 빈도순으로 FG 정렬
    fg_counts = (df[rdkit_props_all] > 0).sum().sort_values(ascending=False)
    
    # 동적으로 제외할 FG 결정 (논문 B.1 기준)
    most_prevalent = fg_counts.head(config.n_most_prevalent).index.tolist()
    rarest = fg_counts.tail(config.n_rarest).index.tolist()
    excluded_fgs = most_prevalent + rarest
    
    # 72개 FG 선택
    retained_fgs = [fg for fg in rdkit_props_all if fg not in excluded_fgs]
    
    print(f"\nFG Selection (논문 B.1 기준):")
    print(f"  - Total FGs: {len(rdkit_props_all)}")
    print(f"  - Excluded: {len(excluded_fgs)} ({config.n_most_prevalent} most prevalent + {config.n_rarest} rarest)")
    print(f"  - Retained (K): {len(retained_fgs)}")
    print(f"\nExcluded FGs:")
    print(f"  Most prevalent ({config.n_most_prevalent}): {most_prevalent}")
    print(f"  Rarest ({config.n_rarest}): {rarest}")
    
    # Sparsity Score 계산 (논문 B.1 수식)
    print("\nComputing sparsity scores (논문 B.1 수식)...")
    eps = 1e-6
    
    # c_g = Σ_i x_{i,g}
    c_g = (df[retained_fgs] > 0).sum(axis=0).values.astype(np.float64)
    
    # s_g = 1 / (c_g + ε)
    s_g = 1.0 / (c_g + eps)
    
    # x_{i,g} ∈ {0, 1}
    x_ig = (df[retained_fgs] > 0).values.astype(np.float64)
    
    # σ_i = (Σ_g x_{i,g} · s_g)²
    sigma_i = (x_ig @ s_g) ** 2
    
    # p_i = σ_i / Σ_j σ_j
    p_i = sigma_i / sigma_i.sum()
    
    # 샘플링
    print(f"\nSampling {config.n_samples:,} molecules...")
    np.random.seed(config.random_seed)
    sampled_indices = np.random.choice(
        len(df), 
        size=config.n_samples, 
        replace=False, 
        p=p_i
    )
    sampled_df = df.iloc[sampled_indices].reset_index(drop=True)
    
    # 저장
    print(f"\nSaving:")
    print(f"  - {config.output_sampled_csv} (all {len(rdkit_props_all)} FGs)")
    sampled_df.to_csv(config.output_sampled_csv, index=False)
    
    print(f"  - {config.output_filtered_csv} ({len(retained_fgs)} FGs for training)")
    # 제외된 FG 컬럼 삭제
    sampled_filtered = sampled_df.drop(columns=excluded_fgs, errors='ignore')
    sampled_filtered.to_csv(config.output_filtered_csv, index=False)
    
    # Figure 6 시각화
    _generate_figure6(df, sampled_df, rdkit_props_all, excluded_fgs, config)
    
    return sampled_df, excluded_fgs, retained_fgs


def _generate_figure6(
    df_full: pd.DataFrame,
    df_sampled: pd.DataFrame,
    rdkit_props_all: List[str],
    excluded_fgs: List[str],
    config: PipelineConfig
) -> None:
    """논문 Figure 6 스타일 시각화 생성"""
    print("\nGenerating Figure 6 visualization...")
    
    # 빈도순 정렬
    before_counts = (df_full[rdkit_props_all] > 0).sum().sort_values(ascending=False)
    fg_order = before_counts.index.tolist()
    
    fig, axes = plt.subplots(3, 1, figsize=(22, 16))
    x = np.arange(len(fg_order))
    
    # === Top Panel: 전체 PubChem 분포 ===
    ax1 = axes[0]
    vals1 = [before_counts[fg] for fg in fg_order]
    colors1 = plt.cm.Purples(np.linspace(0.9, 0.4, len(fg_order)))
    ax1.bar(x, vals1, color=colors1)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(fg_order, rotation=90, fontsize=6)
    ax1.set_xlim(-0.5, len(fg_order) - 0.5)
    ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax1.set_title(f'(a) Full PubChem Distribution ({len(rdkit_props_all)} FGs)', fontsize=12, fontweight='bold')
    
    # === Middle Panel: 제외 후 분포 ===
    ax2 = axes[1]
    vals2 = [0 if fg in excluded_fgs else before_counts[fg] for fg in fg_order]
    colors2 = ['lightgray' if fg in excluded_fgs else colors1[i] for i, fg in enumerate(fg_order)]
    ax2.bar(x, vals2, color=colors2)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(fg_order, rotation=90, fontsize=6)
    ax2.set_xlim(-0.5, len(fg_order) - 0.5)
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    retained_count = len(rdkit_props_all) - len(excluded_fgs)
    ax2.set_title(f'(b) After Excluding {len(excluded_fgs)} FGs (K={retained_count} retained)', 
                  fontsize=12, fontweight='bold')
    
    # === Bottom Panel: Sampling 후 분포 ===
    ax3 = axes[2]
    after_counts = (df_sampled[rdkit_props_all] > 0).sum()
    vals3 = [0 if fg in excluded_fgs else after_counts[fg] for fg in fg_order]
    
    # 보라→초록 그라데이션 (논문 Figure 6 스타일)
    n = len(fg_order)
    colors3 = []
    for i, fg in enumerate(fg_order):
        if fg in excluded_fgs:
            colors3.append('lightgray')
        else:
            ratio = i / n
            if ratio < 0.5:
                colors3.append(plt.cm.Purples(0.9 - ratio * 1.2))
            else:
                colors3.append(plt.cm.Greens(0.3 + (ratio - 0.5) * 1.2))
    
    ax3.bar(x, vals3, color=colors3)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_xlabel('Functional Groups', fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(fg_order, rotation=90, fontsize=6)
    ax3.set_xlim(-0.5, len(fg_order) - 0.5)
    ax3.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax3.set_title(f'(c) After Sparsity-aware Sampling ({config.n_samples:,} molecules)', 
                  fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(config.output_figure, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {config.output_figure}")


# ============================================
# Step 3: SELFIES Conversion + Vocabulary
# ============================================
def _smiles_to_selfies(smiles: str) -> Optional[str]:
    """SMILES → SELFIES 변환 (OpenBabel fallback)"""
    # 1차: 직접 변환
    try:
        selfies = sf.encoder(smiles)
        if selfies is not None:
            return selfies
    except Exception:
        pass
    
    # 2차: RDKit 정규화 후 변환
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical = Chem.MolToSmiles(mol)
            selfies = sf.encoder(canonical)
            if selfies is not None:
                return selfies
    except Exception:
        pass
    
    # 3차: OpenBabel 시도 (설치된 경우)
    try:
        from openbabel import pybel
        mol = pybel.readstring("smi", smiles)
        canonical = mol.write("can").strip()
        selfies = sf.encoder(canonical)
        if selfies is not None:
            return selfies
    except Exception:
        pass
    
    return None


def step3_selfies_vocab(
    df: Optional[pd.DataFrame] = None,
    config: PipelineConfig = CONFIG
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Step 3: SELFIES 변환 + Vocabulary 구축
    
    논문 B.2: "For SELFIES reconstruction, sequences are truncated to max length of 512 tokens"
    
    Vocab 구조 (train 코드와 동일):
    - 0: <pad>
    - 1: <unk>
    - 2: <eos>
    - 3: .
    - 4+: sorted SELFIES tokens
    """
    print("\n" + "=" * 70)
    print("Step 3: SELFIES Conversion + Vocabulary")
    print("=" * 70)
    
    if df is None:
        print(f"Loading {config.output_filtered_csv}...")
        df = pd.read_csv(config.output_filtered_csv)
    
    print(f"Total molecules: {len(df):,}")
    
    # SELFIES 변환 (멀티프로세싱)
    print(f"\nConverting to SELFIES using {config.n_cores} cores...")
    smiles_list = df['SMILES'].tolist()
    
    with Pool(config.n_cores) as pool:
        selfies_list = list(tqdm(
            pool.imap(_smiles_to_selfies, smiles_list, chunksize=2000),
            total=len(smiles_list),
            desc="SMILES → SELFIES"
        ))
    
    # 유효한 것만 필터링
    valid_mask = [s is not None for s in selfies_list]
    valid_count = sum(valid_mask)
    print(f"Valid: {valid_count:,} / {len(df):,} ({100 * valid_count / len(df):.2f}%)")
    
    df = df[valid_mask].reset_index(drop=True)
    selfies_list = [s for s in selfies_list if s is not None]
    df['SELFIES'] = selfies_list
    
    # Vocabulary 구축
    print("\nBuilding vocabulary...")
    all_tokens = set()
    for selfies in tqdm(selfies_list, desc="Extracting tokens"):
        tokens = list(sf.split_selfies(selfies))
        all_tokens.update(tokens)
    
    # 특수 토큰 + 정렬된 토큰 (train 코드와 동일한 구조)
    special_tokens = ['<pad>', '<unk>', '<eos>', '.']
    for tok in special_tokens:
        if tok in all_tokens:
            all_tokens.remove(tok)
    vocab_list = special_tokens + sorted(list(all_tokens))
    vocab_dict = {tok: idx for idx, tok in enumerate(vocab_list)}
    
    print(f"Vocabulary size: {len(vocab_dict)}")
    print(f"Special tokens: {special_tokens} → indices {list(range(len(special_tokens)))}")
    
    # SELFIES 길이 통계
    selfies_lengths = [len(list(sf.split_selfies(s))) for s in selfies_list[:10000]]
    print(f"\nSELFIES length stats (first 10k):")
    print(f"  Mean: {np.mean(selfies_lengths):.1f}")
    print(f"  Max:  {max(selfies_lengths)}")
    print(f"  Min:  {min(selfies_lengths)}")
    
    # 저장
    print(f"\nSaving:")
    print(f"  - {config.output_selfies_csv}")
    df.to_csv(config.output_selfies_csv, index=False)
    
    print(f"  - {config.output_vocab_json}")
    with open(config.output_vocab_json, 'w') as f:
        json.dump(vocab_dict, f, indent=2)
    
    return df, vocab_dict


# ============================================
# Step 4: PyTorch Geometric Graph Conversion
# ============================================
# Multiprocessing을 위한 전역 변수
_g_vocab_dict: Optional[Dict[str, int]] = None
_g_fg_columns: Optional[List[str]] = None
_g_max_seq_len: int = 512


def _init_graph_worker(vocab: Dict[str, int], fg_cols: List[str], max_len: int) -> None:
    """Worker 초기화"""
    global _g_vocab_dict, _g_fg_columns, _g_max_seq_len
    _g_vocab_dict = vocab
    _g_fg_columns = fg_cols
    _g_max_seq_len = max_len


def _process_molecule_to_graph(args: Tuple[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """단일 분자 → Graph dict (pickle 가능)"""
    idx, row_dict = args
    
    try:
        smiles = row_dict['SMILES']
        selfies = row_dict['SELFIES']
        
        # Graph 생성
        graph = smiles2graph(smiles)
        
        n_nodes = graph['node_feat'].shape[0]
        n_edges = graph['edge_index'].shape[1]
        
        # 크기 제한 (논문 B.2: max_seq_len=512)
        if n_nodes > _g_max_seq_len or n_edges > _g_max_seq_len * 2:
            return None
        
        # SELFIES 토큰화
        tokens = list(sf.split_selfies(selfies))
        token_ids = [_g_vocab_dict.get(t, _g_vocab_dict['<unk>']) for t in tokens]
        token_ids.append(_g_vocab_dict['<eos>'])
        
        if len(token_ids) > _g_max_seq_len:
            return None
        
        # FG vector (72개, binary)
        fg_vector = [1 if row_dict[fg] > 0 else 0 for fg in _g_fg_columns]
        
        return {
            'x': graph['node_feat'],
            'edge_index': graph['edge_index'],
            'edge_attr': graph['edge_feat'],
            'fg_vector': fg_vector,
            'selfies_tokens': token_ids
        }
    except Exception:
        return None
def step4_graph_conversion(
    df: Optional[pd.DataFrame] = None,
    vocab_dict: Optional[Dict[str, int]] = None,
    config: PipelineConfig = CONFIG,
    # chunk_size는 이제 사용하지 않음 (메모리 방식)
) -> None:
    """
    [최종 수정] Step 4: 그래프 변환 후 'Sparsity Sampling'을 적용하여 Train/Valid 분리 저장
    - Input: 약 5.3M개 분자 (Step 2 결과)
    - Logic: 전체 변환 -> Sparsity Score 계산 -> 5M Train (Weighted) / 2k Valid (Rest)
    - Output: Collated Format (.pt) for H100 Speed
    """
    print("\n" + "=" * 70)
    print("Step 4: PyTorch Geometric Graph Conversion & Distribution Splitting")
    print("=" * 70)
    
    if df is None:
        print(f"Loading {config.output_selfies_csv}...")
        df = pd.read_csv(config.output_selfies_csv)
    
    if vocab_dict is None:
        print(f"Loading {config.output_vocab_json}...")
        with open(config.output_vocab_json, 'r') as f:
            vocab_dict = json.load(f)
    
    fg_columns = [c for c in df.columns if c.startswith('fr_')]
    print(f"Total molecules to process: {len(df):,}")
    
    # 1. 멀티프로세싱으로 전체 그래프 변환 (List[Data] 생성)
    row_dicts = df.to_dict('records')
    args_list = list(enumerate(row_dicts))
    
    print(f"\n[1/4] Converting all molecules to graphs ({config.n_cores} cores)...")
    all_graphs = []
    
    with Pool(
        config.n_cores,
        initializer=_init_graph_worker,
        initargs=(vocab_dict, fg_columns, config.max_seq_len)
    ) as pool:
        results = list(tqdm(
            pool.imap(_process_molecule_to_graph, args_list, chunksize=2000),
            total=len(args_list),
            desc="Building graphs"
        ))
        
    # 유효한 그래프만 수집
    for r in results:
        if r is not None:
            all_graphs.append(Data(
                x=torch.tensor(r['x'], dtype=torch.long),
                edge_index=torch.tensor(r['edge_index'], dtype=torch.long),
                edge_attr=torch.tensor(r['edge_attr'], dtype=torch.long),
                fg_vector=torch.tensor(r['fg_vector'], dtype=torch.float32),
                selfies_tokens=torch.tensor(r['selfies_tokens'], dtype=torch.long)
            ))
            
    total_len = len(all_graphs)
    print(f"Valid graphs converted: {total_len:,}")
    
    # 2. Sparsity Score 재계산 (분포 유지를 위해 필수)
    print("\n[2/4] Calculating Sparsity Scores for Splitting...")
    
    # FG Matrix 생성 (N, 72)
    fg_matrix = np.zeros((total_len, len(fg_columns)), dtype=np.float32)
    for i, g in enumerate(tqdm(all_graphs, desc="Extracting FG Vectors")):
        fg_matrix[i] = g.fg_vector.numpy()
        
    epsilon = 1e-6
    c_g = fg_matrix.sum(axis=0)          # Column sum
    s_g = 1.0 / (c_g + epsilon)          # Sparsity factor
    score_i = (fg_matrix @ s_g) ** 2     # Molecule score
    p_i = score_i / score_i.sum()        # Probability
    
    # 3. 샘플링 (Train 5M + Valid 2k)
    TRAIN_SIZE = 5_000_000
    VALID_SIZE = 2_000
    
    print(f"\n[3/4] Sampling: Train({TRAIN_SIZE:,}) + Valid({VALID_SIZE:,}) from Rest")
    
    if total_len < TRAIN_SIZE + VALID_SIZE:
        raise ValueError(f"Not enough data! Have {total_len}, need {TRAIN_SIZE + VALID_SIZE}")

    rng = np.random.default_rng(seed=42)
    
    # Train: 가중치(p_i)를 적용하여 뽑음 -> Step 2 분포 유지
    train_indices = rng.choice(total_len, size=TRAIN_SIZE, replace=False, p=p_i)
    train_set = set(train_indices)
    
    # Rest: Train에 안 뽑힌 애들 찾기
    rest_indices = np.array([i for i in range(total_len) if i not in train_set])
    
    # Valid: 남은 것 중에서 2000개 랜덤 추출
    valid_indices = rng.choice(rest_indices, size=VALID_SIZE, replace=False)
    
    # 리스트 분리
    train_graphs = [all_graphs[i] for i in train_indices]
    valid_graphs = [all_graphs[i] for i in valid_indices]
    
    del all_graphs # 메모리 확보
    
    # 4. Collated Format 변환 및 저장
    print("\n[4/4] Collating and Saving (Optimized Format)...")
    
    # Train 저장
    print(f" -> Collating Train ({len(train_graphs):,})...")
    data_train, slices_train = InMemoryDataset.collate(train_graphs)
    train_path = config.data_dir / "graphs_5M_final.pt"
    torch.save((data_train, slices_train), train_path)
    print(f"    Saved: {train_path}")
    
    # Valid 저장
    print(f" -> Collating Valid ({len(valid_graphs):,})...")
    data_valid, slices_valid = InMemoryDataset.collate(valid_graphs)
    valid_path = config.data_dir / "graphs_valid_2k.pt"
    torch.save((data_valid, slices_valid), valid_path)
    print(f"    Saved: {valid_path}")
    
    print("\nDone! Use 'graphs_5M_final.pt' and 'graphs_valid_2k.pt' for training.")
    list_out_path = config.data_dir / "graphs_5M_list.pt"
    tmp_dataset = _CollatedDataset(data_train, slices_train, max_seq_len=config.max_seq_len)
    graphs_list = [tmp_dataset.get(i) for i in range(len(tmp_dataset))]
    torch.save(graphs_list, list_out_path)
    print(f"    Saved List[Data]: {list_out_path}")




# ============================================
# Verification
# ============================================
def verify_pipeline(config: PipelineConfig = CONFIG) -> None:
    """파이프라인 결과 검증"""
    print("\n" + "=" * 70)
    print("Pipeline Verification")
    print("=" * 70)
    
    errors = []
    
    # 1. Filtered CSV 검증
    print("\n[1] Checking filtered CSV...")
    try:
        df = pd.read_csv(config.output_filtered_csv, nrows=100)
        fg_columns = [c for c in df.columns if c.startswith('fr_')]
        print(f"  ✓ FG columns: {len(fg_columns)} (expected: 72)")
        if len(fg_columns) != 72:
            errors.append(f"FG column count mismatch: {len(fg_columns)} != 72")
    except Exception as e:
        errors.append(f"Failed to load filtered CSV: {e}")
    
    # 2. Vocab 검증
    print("\n[2] Checking vocabulary...")
    try:
        with open(config.output_vocab_json, 'r') as f:
            vocab = json.load(f)
        
        expected = {'<pad>': 0, '<unk>': 1, '<eos>': 2, '.': 3}
        for tok, idx in expected.items():
            if vocab.get(tok) != idx:
                errors.append(f"Vocab mismatch: {tok} should be {idx}, got {vocab.get(tok)}")
            else:
                print(f"  ✓ vocab['{tok}'] = {idx}")
        
        print(f"  ✓ Total vocab size: {len(vocab)}")
    except Exception as e:
        errors.append(f"Failed to load vocab: {e}")
    
    # 3. Graphs 검증
    print("\n[3] Checking graphs...")
    try:
        graphs = torch.load(config.output_graphs_train_pt,weights_only=False)
        print(f"  ✓ Total graphs: {len(graphs):,}")
        
        # 첫 번째 그래프 검증
        g = graphs[0]
        assert g.fg_vector.shape[0] == len(fg_columns), \
            f"fg_vector dim mismatch: {g.fg_vector.shape[0]} != {len(fg_columns)}"
        print(f"  ✓ fg_vector.shape: {g.fg_vector.shape}")
        print(f"  ✓ x.shape: {g.x.shape}")
        print(f"  ✓ edge_index.shape: {g.edge_index.shape}")
        
        # selfies_tokens 검증
        assert g.selfies_tokens[-1].item() == vocab['<eos>'], \
            "Last token should be <eos>"
        print(f"  ✓ selfies_tokens ends with <eos>")
        
    except Exception as e:
        errors.append(f"Failed to verify graphs: {e}")
    
    # 결과 출력
    print("\n" + "=" * 70)
    if errors:
        print("❌ Verification FAILED:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("✅ All verifications passed!")
    print("=" * 70)


# ============================================
# Main Pipeline
# ============================================
def run_pipeline(
    steps: Optional[List[int]] = None,
    config: PipelineConfig = CONFIG
) -> None:
    """
    전체 파이프라인 실행
    
    Args:
        steps: 실행할 step 리스트 (None이면 전체 실행)
        config: 파이프라인 설정
    """
    if steps is None:
        steps = [1, 2, 3, 4]
    
    print("=" * 70)
    print("Mol-LLM Stage 1: Data Preparation Pipeline v3")
    print("=" * 70)
    print(f"Data directory: {config.data_dir}")
    print(f"Steps to run: {steps}")
    print(f"N samples: {config.n_samples:,}")
    print(f"Max seq len: {config.max_seq_len}")
    print(f"N cores: {config.n_cores}")
    
    df = None
    vocab_dict = None
    
    if 1 in steps:
        df = step1_compute_fg(config)
    
    if 2 in steps:
        df, excluded_fgs, retained_fgs = step2_sparsity_sampling(df, config)
        # Step 2 이후에는 filtered CSV 사용
        df = pd.read_csv(config.output_filtered_csv)
    
    if 3 in steps:
        df, vocab_dict = step3_selfies_vocab(df, config)
    
    if 4 in steps:
        step4_graph_conversion(df, vocab_dict, config)
    
    # 검증
    # verify_pipeline(config)
    
    print("\n" + "=" * 70)
    print("Pipeline completed!")
    print("=" * 70)
    print("\nOutput files:")
    print(f"  - {config.output_fg_csv}")
    print(f"  - {config.output_sampled_csv}")
    print(f"  - {config.output_filtered_csv}")
    print(f"  - {config.output_selfies_csv}")
    print(f"  - {config.output_vocab_json}")
    print(f"  - {config.output_graphs_pt}")
    print(f"  - {config.output_figure}")


# ============================================
# Entry Point
# ============================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Mol-LLM Stage 1: Data Preparation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_pipeline_v3.py                    # Run all steps
  python data_pipeline_v3.py --step 1           # Run only step 1
  python data_pipeline_v3.py --step 2 3 4       # Run steps 2, 3, 4
  python data_pipeline_v3.py --verify           # Only verify outputs
  python data_pipeline_v3.py --n-samples 100000 # Custom sample size
        """
    )
    parser.add_argument(
        '--step', 
        type=int, 
        nargs='+',
        default=None,
        help='Steps to run (1-4). Default: all steps'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only run verification'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=5_300_000,
        help='Number of molecules to sample (default: 5M)'
    )
    parser.add_argument(
        '--n-cores',
        type=int,
        default=64,
        help='Number of CPU cores (default: 64)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Data directory path'
    )
    
    args = parser.parse_args()
    
    # Config 업데이트
    config = PipelineConfig()
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    config.n_samples = args.n_samples
    config.n_cores = args.n_cores
    
    if args.verify:
        verify_pipeline(config)
    else:
        run_pipeline(steps=args.step, config=config)