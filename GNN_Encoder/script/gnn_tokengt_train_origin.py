from ast import In
import os, sys
from pathlib import Path 
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # GNN_Encoder 디렉토리
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import json
import torch
import pickle
import selfies as sf
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import warnings

from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, GPT2Config
from model.tokenGT import TokenGT, BERTTokenGT
from sklearn.exceptions import UndefinedMetricWarning
from rdkit import Chem
from rdkit.Chem import Fragments
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

# ★ PyG DataLoader 사용 (원작자 방식)
from torch_geometric.data import DataLoader, Data

from ogb.utils import smiles2graph
from torch_geometric.datasets import QM9
from model.gin_model import GNN, GNN_MoleculeSTM
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from tqdm import tqdm
from rdkit import RDLogger
from sklearn.metrics import accuracy_score, roc_auc_score

from pytorch_lightning.loggers import WandbLogger
import wandb

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
torch.set_float32_matmul_precision('medium')

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)


DATA_DIR = PROJECT_ROOT / "data"

SELFIES_VOCAB_PATH = DATA_DIR / "selfies_vocab.json"
PUBCHEM_CSV_PATH = DATA_DIR / "pubchem_5M_sampled_filtered.csv"
GRAPHS_PATH = DATA_DIR / "graphs_5M_list.pt"

RDKIT_PROPS = ['fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
               'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2',
               'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
               'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
               'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
               'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
               'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
               'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester',
               'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
               'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
               'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine',
               'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
               'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
               'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine',
               'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
               'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
               'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']
RDKIT_PROPS = RDKIT_PROPS[12:-1]

func_idx2name = {idx: name for idx, name in enumerate(RDKIT_PROPS)}

qm9_name2idx = {
    "dipole": 0,
    "isotropic": 1,
    "homo": 2,
    "lumo": 3,
    "gap": 4,
    "electronic": 5,
    "vibrational": 6,
    "internalEnergy0K": 7,
    "internalEnergy298K": 8,
    "enthalpy": 9,
    "freeEnergy298K": 10,
    "capavity": 11,
    "atomizationEnergy0K": 12,
    "atomizationEnergy298K": 13,
    "atomizationEnthalpy298K": 14,
    "atomizationFreeEnergy298K": 15,
    "rotationA": 16,
    "rotationB": 17,
    "rotationC": 18,
}

model_settings = {
    # "TokenGT": {
    #     "backbone": "TokenGT->MLP",
    #     "activate_GNN": True,
    #     "activate_MLP": True,
    #     "lr": 1e-4,
    #     "max_seq_len": 512,
    #     "gnn_output_dim": 1024,
    # }
    "MoleculeSTM": {
        "backbone": "MoleculeSTM->MLP",
        "activate_GNN": True,
        "activate_MLP": True,
        "lr": 1e-4,
        "max_seq_len": 512,
        "gnn_output_dim": 1024,
    },
}


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor, mask=None):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class SelfiesTokenizer:
    def __init__(self):
        if not SELFIES_VOCAB_PATH.exists():
            df = pd.read_csv(PUBCHEM_CSV_PATH)
            selfies_tokens = set()
            for i, row in tqdm(enumerate(df.iterrows()), total=len(df), desc="Extracting SELFIES tokens"):
                smiles = row[1]['SMILES']
                try:
                    selfies = sf.encoder(smiles)
                except:
                    continue
                selfies_tokens.update(sf.get_alphabet_from_selfies(list(sf.split_selfies(selfies))))
            self.vocab = sorted(list(selfies_tokens))
            selfies_vocab = {token: idx for idx, token in enumerate(self.vocab)}
            with open(SELFIES_VOCAB_PATH, "w") as f:
                json.dump(selfies_vocab, f, indent=4)
            self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
            self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        else:
            with open(SELFIES_VOCAB_PATH, "r") as f:
                selfies_vocab = json.load(f)
            self.vocab = sorted(list(selfies_vocab.keys()))
            self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
            self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        
        # Add "." token in the front
        self.vocab = ["."] + self.vocab
        # Add special tokens in the front
        self.vocab = ["<pad>", "<unk>", "<eos>"] + self.vocab
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def tokenize(self, selfies_str):
        return sf.split_selfies(selfies_str)

    def encode(self, selfies_str):
        return [self.token_to_id[tok] for tok in self.tokenize(selfies_str)] + [self.token_to_id["<eos>"]]

    def decode(self, token_ids):
        return sf.decoder("".join([self.id_to_token[i] for i in token_ids]))

    def vocab_size(self):
        return len(self.vocab)


# ============================================================
# ★ 원작자 방식 DataModule (List[Data] + PyG DataLoader)
# ============================================================
class DataModule_Optimized(pl.LightningDataModule):
    """
    원작자 방식: List[Data]를 직접 로드하고 PyG DataLoader 사용
    - custom_collate 없음
    - 빠른 학습 속도
    """
    def __init__(self, tokenizer, batch_size=64, max_seq_len=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        
        # ★ List[Data] 형태 파일 경로
        self.graphs_path = GRAPHS_PATH

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        print(f"📥 Loading graphs from {self.graphs_path}...")
        print("   (This may take 20-30 minutes for 5M molecules)")
        
        # ★ List[Data] 직접 로드 (원작자 방식)
        graphs = torch.load(self.graphs_path, weights_only=False)
        
        print(f"✅ Loaded {len(graphs):,} graphs")
        
        # 길이 필터링 (원작자 코드와 동일)
        filtered_graphs = []
        total_skipped = 0
        
        for graph in tqdm(graphs, desc="Filtering by seq_len"):
            seq_len = graph.x.shape[0] + graph.edge_index.shape[1] + 1
            if seq_len > self.max_seq_len:
                total_skipped += 1
                continue
            
            # selfies_tokens를 리스트로 변환 (원작자 forward 호환)
            if isinstance(graph.selfies_tokens, torch.Tensor):
                graph.selfies_tokens = graph.selfies_tokens.tolist()[:self.max_seq_len]
            else:
                graph.selfies_tokens = graph.selfies_tokens[:self.max_seq_len]
            
            filtered_graphs.append(graph)
        
        print(f"⚠️  Skipped {total_skipped:,} graphs (seq_len > {self.max_seq_len})")
        print(f"✅ Final dataset: {len(filtered_graphs):,} graphs")
        
        del graphs  # 메모리 해제
        
        # Train/Val/Test 분할 (원작자와 동일)
        val_size = 3000
        test_size = 3000
        
        self.train_set = filtered_graphs[:-val_size-test_size]
        self.val_set = filtered_graphs[-val_size-test_size:-test_size]
        self.test_set = filtered_graphs[-test_size:]
        
        print(f"📊 Split: Train={len(self.train_set):,}, Val={len(self.val_set):,}, Test={len(self.test_set):,}")

    def train_dataloader(self):
        # ★ PyG DataLoader 사용 (원작자 방식, collate_fn 없음!)
        return DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0,  # PyG는 num_workers=0이 안정적
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True
        )


# ============================================================
# TaskModel (원작자 forward 방식 - pad_sequence 사용)
# ============================================================
class TaskModel(pl.LightningModule):
    def __init__(self, tokenizer, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = kwargs['max_seq_len']
        self.gnn_output_dim = kwargs['gnn_output_dim']
        self.backbone = kwargs['backbone']
        self.activate_GNN = kwargs['activate_GNN']
        self.activate_MLP = kwargs['activate_MLP']
        self.lr = kwargs['lr']

        self.n_funcgroup = len(RDKIT_PROPS)

        ################## Initialize GNN ##################
        if self.backbone == "MoleculeSTM->MLP":
            self.gnn = GNN_MoleculeSTM(
                num_layer=5,
                emb_dim=self.gnn_output_dim,
                gnn_type="gin",
                drop_ratio=0.0,
                JK="last",
            )
        elif self.backbone == "TokenGT->MLP":
            self.gnn = BERTTokenGT(
                input_feat_dim=9,
                hidden_dim=self.gnn_output_dim,
                num_layers=5,
                num_heads=8,
                method="laplacian",
                d_p=64,
                d_e=64,
                use_graph_token=True,
                max_position_embeddings=1024
            )
        else:
            raise ValueError(f"Invalid backbone type: {self.backbone}")
        
        self.ln_graph = LayerNorm(self.gnn_output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.gnn_output_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_funcgroup)
        )

        pad_id = tokenizer.token_to_id["<pad>"]
        eos_id = tokenizer.token_to_id["<eos>"]
        self.pad_id = pad_id
        self.eos_id = eos_id

        config = GPT2Config(
            vocab_size=tokenizer.vocab_size(),
            n_positions=self.max_seq_len,
            n_embd=self.gnn_output_dim,
            n_layer=6,
            n_head=8,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            attn_implementation="flash_attention_2",
        )
        self.selfies_decoder = GPT2LMHeadModel(config)
        self.tokenizer = tokenizer

    def forward(self, x, edge_index, edge_attr, batch, selfies_tokens):
        """
        ★ 원작자 방식 forward (selfies_tokens가 List[List[int]])
        """
        output, _ = self.gnn(x, edge_index, edge_attr, batch)
        graph_embedding = output[:, :1, :]
        graph_embedding = self.ln_graph(graph_embedding)
        pred_funcgroup = self.mlp(graph_embedding.squeeze(1))

        # ★ 원작자 방식: 리스트에서 패딩 (forward 내에서 처리)
        pad_id = self.pad_id
        input_ids_tensor = [torch.tensor(ids, dtype=torch.long) for ids in selfies_tokens]
        padded_input_ids = pad_sequence(input_ids_tensor, batch_first=True, padding_value=pad_id).to(graph_embedding.device)
        
        # Shift for decoder input and labels
        decoder_input_ids = padded_input_ids[:, :-1]
        labels = padded_input_ids.clone()
        attention_mask = (decoder_input_ids != pad_id).long()

        # Embed tokens
        token_embeds = self.selfies_decoder.transformer.wte(decoder_input_ids)
        inputs_embeds = torch.cat([graph_embedding, token_embeds], dim=1)

        # Adjust attention mask
        extended_attention_mask = torch.cat([
            torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device),
            attention_mask
        ], dim=1)

        labels[labels == pad_id] = -100

        # cut off by max_seq_len
        inputs_embeds = inputs_embeds[:, :self.max_seq_len, :]
        extended_attention_mask = extended_attention_mask[:, :self.max_seq_len]
        labels = labels[:, :self.max_seq_len]
        
        # Decode
        pred_selfies = self.selfies_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=labels
        )

        return pred_funcgroup, pred_selfies

    def training_step(self, batch, batch_idx):
        pred_funcgroup, pred_selfies = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.selfies_tokens)
        pred_funcgroup = torch.sigmoid(pred_funcgroup)
        task_size = pred_funcgroup.size(1)
        y = batch.fg_vector.view(-1, task_size).float()
        loss_funcgroup = F.binary_cross_entropy_with_logits(pred_funcgroup, y.float())
        self.log('train/loss_funcgroup', loss_funcgroup, prog_bar=False, sync_dist=True, batch_size=batch.fg_vector.size(0))
        self.log('train/loss_selfies', pred_selfies.loss, prog_bar=False, sync_dist=True, batch_size=batch.fg_vector.size(0))
        total_loss = loss_funcgroup + pred_selfies.loss
        self.log('train/total_loss', total_loss, prog_bar=True, sync_dist=True, batch_size=batch.fg_vector.size(0))

        return total_loss

    def validation_step(self, batch, batch_idx):
        pred_funcgroup, pred_selfies = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.selfies_tokens)
        pred_funcgroup = torch.sigmoid(pred_funcgroup)
        task_size = pred_funcgroup.size(1)
        y = batch.fg_vector.view(-1, task_size).float()
        loss_funcgroup = F.binary_cross_entropy_with_logits(pred_funcgroup, y.float())
        self.log('validation/loss_funcgroup', loss_funcgroup, prog_bar=False, sync_dist=True, batch_size=batch.fg_vector.size(0))
        self.log('validation/loss_selfies', pred_selfies.loss, prog_bar=False, sync_dist=True, batch_size=batch.fg_vector.size(0))
        self.log('validation/total_loss', loss_funcgroup + pred_selfies.loss, prog_bar=True, sync_dist=True, batch_size=batch.fg_vector.size(0))

        return_dict = {
            "pred_funcgroup": pred_funcgroup.detach(),
            "y_funcgroup": y.detach(),
            "loss_funcgroup": loss_funcgroup.detach(),
            "loss_selfies": pred_selfies.loss.detach(),
        }
        return return_dict

    def test_step(self, batch, batch_idx):
        pred_funcgroup, pred_selfies = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.selfies_tokens)
        pred_funcgroup = torch.sigmoid(pred_funcgroup)
        task_size = pred_funcgroup.size(1)
        y = batch.fg_vector.view(-1, task_size).float()
        loss_funcgroup = F.binary_cross_entropy_with_logits(pred_funcgroup, y.float())
        self.log('test/loss_funcgroup', loss_funcgroup, prog_bar=False, sync_dist=True, batch_size=batch.fg_vector.size(0))
        self.log('test/loss_selfies', pred_selfies.loss, prog_bar=False, sync_dist=True, batch_size=batch.fg_vector.size(0))

        return_dict = {
            "pred_funcgroup": pred_funcgroup.detach(),
            "y_funcgroup": y.detach(),
            "loss_funcgroup": loss_funcgroup.detach(),
            "loss_selfies": pred_selfies.loss.detach(),
        }
        return return_dict

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_train_epoch_start(self):
        for name, param in self.gnn.state_dict().items():
            try:
                self.logger.log_metrics({f"GNN_parameters/{name}_mean": param.mean()}, step=self.current_epoch)
            except:
                self.logger.log_metrics({f"GNN_parameters/{name}": param}, step=self.current_epoch)
        for name, param in self.mlp.state_dict().items():
            try:
                self.logger.log_metrics({f"MLP_parameters/{name}_mean": param.float().mean()}, step=self.current_epoch)
            except:
                self.logger.log_metrics({f"MLP_parameters/{name}": param}, step=self.current_epoch)
        self.logger.log_metrics({'hyperparameters/lr': self.lr}, step=self.current_epoch)


class ValidationCallback(Callback):
    def __init__(self):
        self.val_outs = []
        self.val_labels = []
        self.val_loss = []
        self.test_outs = []
        self.test_labels = []
        self.test_loss = []
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.val_outs.append(outputs['pred_funcgroup'].detach().cpu())
        self.val_labels.append(outputs['y_funcgroup'].detach().cpu())

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.test_outs.append(outputs['pred_funcgroup'].detach().cpu())
        self.test_labels.append(outputs['y_funcgroup'].detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        preds = torch.cat(self.val_outs, dim=0)
        ys = torch.cat(self.val_labels, dim=0)
        preds = pl_module.all_gather(preds)
        ys = pl_module.all_gather(ys)
        preds = preds.view(-1, preds.size(-1))
        ys = ys.view(-1, ys.size(-1))
        if trainer.is_global_zero:
            for i in range(preds.size(1)):
                pred = preds[:, i]
                y = ys[:, i]
                acc = (pred > 0.5).float() == y
                acc = acc.float().mean().item()
                auroc = roc_auc_score(y.cpu().float().numpy(), pred.cpu().float().numpy())
                trainer.logger.log_metrics({f'validation_acc/{func_idx2name[i]}': acc}, step=trainer.current_epoch)
                trainer.logger.log_metrics({f'validation_auroc/{func_idx2name[i]}': auroc}, step=trainer.current_epoch)
                trainer.logger.log_metrics({f'validation_num/{func_idx2name[i]}': y.sum().item()}, step=trainer.current_epoch)
            trainer.logger.log_metrics({'validation/instance_size': preds.size(0)}, step=trainer.current_epoch)
        self.val_outs = []
        self.val_labels = []
    
    def on_test_epoch_end(self, trainer, pl_module):
        preds = torch.cat(self.test_outs, dim=0)
        ys = torch.cat(self.test_labels, dim=0)
        preds = pl_module.all_gather(preds)
        ys = pl_module.all_gather(ys)
        preds = preds.view(-1, preds.size(-1))
        ys = ys.view(-1, ys.size(-1))
        if trainer.is_global_zero:
            for i in range(preds.size(1)):
                pred = preds[:, i]
                y = ys[:, i]
                acc = (pred > 0.5).float() == y
                acc = acc.float().mean().item()
                auroc = roc_auc_score(y.cpu().float().numpy(), pred.cpu().float().numpy())
                trainer.logger.log_metrics({f'test_acc/{func_idx2name[i]}': acc}, step=trainer.current_epoch)
                trainer.logger.log_metrics({f'test_auroc/{func_idx2name[i]}': auroc}, step=trainer.current_epoch)
                trainer.logger.log_metrics({f'test_num/{func_idx2name[i]}': y.sum().item()}, step=trainer.current_epoch)
            trainer.logger.log_metrics({'test/instance_size': preds.size(0)}, step=trainer.current_epoch)
        self.test_outs = []
        self.test_labels = []


# ============================================================
# 학습 실행
# ============================================================
if __name__ == '__main__':

    model_types = list(model_settings.keys())

    checkpoint_callback = ModelCheckpoint(
        monitor='validation/total_loss',
        mode='min',
        save_top_k=3,
        filename='epoch{epoch:02d}-step{step:06d}'
    )
    validation_callback = ValidationCallback()
    tokenizer = SelfiesTokenizer()

    for model_type in model_types:
        # ★ 최적화된 DataModule 사용
        data_module = DataModule_Optimized(
            tokenizer, 
            batch_size=64, 
            max_seq_len=model_settings[model_type]['max_seq_len']
        )
        
        pl_model = TaskModel(tokenizer=tokenizer, **model_settings[model_type])
        
        WANDB_PROJECT = f"model_type_{model_type}-mol-llm-gnn-pretraining"
        WANDB_RUN_NAME = f"H100-8GPU-OPTIMIZED-{model_type}"
        wandb_logger = WandbLogger(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            save_dir='./wandb_logs',
            config=model_settings[model_type]
        )
        
        trainer = pl.Trainer(
            max_epochs=50,
            accelerator='gpu',
            devices=[0,1,6,7],
            default_root_dir=f'./gnn_ablation/custom_gnn_train_5M/{model_type}',
            callbacks=[checkpoint_callback, validation_callback],
            strategy='ddp',            
            precision="bf16-mixed",
            gradient_clip_val=0.5,
            log_every_n_steps=600,
            val_check_interval=0.25,
            num_sanity_val_steps=0,
            logger=wandb_logger
        )
        
        trainer.fit(pl_model, datamodule=data_module)
        
        wandb.finish()