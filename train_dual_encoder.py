import os
import math
import requests
import time
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel


# ===================== 配置 =====================

@dataclass
class Config:
    # 路径
    data_path: str = "/home/daicheng/bxsong_homework_protein/dataset/train.parquet"
    chemberta_path: str = (
        "/home/daicheng/bxsong_homework_protein/pretrained_model/chemBERTa/chemBERTa_pretrained_model"
    )
    saprot_path: str = (
        "/home/daicheng/bxsong_homework_protein/pretrained_model/SaProt/SaProt"
    )
    output_dir: str = "/home/daicheng/bxsong_homework_protein/dual_encoder_ckpt"
    protein_sequence_cache: str = "/home/daicheng/bxsong_homework_protein/protein_sequences_cache.csv"

    # 训练超参
    batch_size: int = 256
    num_epochs: int = 150
    lr: float = 1e-3           # 提高学习率
    weight_decay: float = 0.001
    max_len_mol: int = 256
    max_len_prot: int = 1024
    projection_dim: int = 1024
    temperature: float = 0.1   # 提高温度，使梯度更平滑
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 改进的训练策略
    use_mean_pooling: bool = True  # 使用平均池化而不是CLS
    gradient_clip: float = 1.0     # 梯度裁剪
    warmup_epochs: int = 5         # 学习率预热


cfg = Config()


# ===================== UniProt API 获取序列 =====================

def fetch_protein_sequence(uniprot_id: str, cache: Dict[str, str]) -> Optional[str]:
    """
    从UniProt API获取蛋白质序列，带缓存
    """
    if uniprot_id in cache:
        return cache[uniprot_id]
    
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # 解析FASTA格式
            lines = response.text.strip().split('\n')
            sequence = ''.join(lines[1:])  # 跳过第一行header
            cache[uniprot_id] = sequence
            time.sleep(0.1)  # 避免请求过快
            return sequence
        else:
            print(f"Warning: Failed to fetch {uniprot_id}, status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching {uniprot_id}: {e}")
        return None


def load_or_fetch_sequences(df: pd.DataFrame, cache_file: str) -> pd.DataFrame:
    """
    加载或获取蛋白质序列，使用缓存避免重复请求
    """
    # 加载已有缓存
    sequence_cache = {}
    if os.path.exists(cache_file):
        cache_df = pd.read_csv(cache_file)
        sequence_cache = dict(zip(cache_df['uniprot_id'], cache_df['sequence']))
        print(f"Loaded {len(sequence_cache)} sequences from cache")
    
    # 获取缺失的序列
    unique_uniprot_ids = df['target__uniprot_id'].unique()
    new_sequences = {}
    
    for uniprot_id in unique_uniprot_ids:
        if uniprot_id not in sequence_cache:
            seq = fetch_protein_sequence(uniprot_id, sequence_cache)
            if seq:
                new_sequences[uniprot_id] = seq
                sequence_cache[uniprot_id] = seq
    
    # 保存更新的缓存
    if new_sequences:
        all_cache = {**sequence_cache, **new_sequences}
        cache_df = pd.DataFrame({
            'uniprot_id': list(all_cache.keys()),
            'sequence': list(all_cache.values())
        })
        cache_df.to_csv(cache_file, index=False)
        print(f"Saved {len(new_sequences)} new sequences to cache")
    
    # 将序列添加到DataFrame
    df['protein_sequence'] = df['target__uniprot_id'].map(sequence_cache)
    
    # 过滤掉没有序列的样本
    df = df.dropna(subset=['protein_sequence']).reset_index(drop=True)
    print(f"After adding sequences: {len(df)} samples")
    
    return df


# ===================== 数据集定义 =====================

class DrugTargetDataset(Dataset):
    def __init__(self, df):
        # 只保留正样本
        if "outcome_is_active" in df.columns:
            df = df[df["outcome_is_active"].astype(bool)].reset_index(drop=True)
        
        # 确保有protein_sequence列
        if 'protein_sequence' not in df.columns:
            print("Warning: protein_sequence not found, using uniprot_id as fallback")
            df['protein_sequence'] = df['target__uniprot_id']
        
        # 过滤掉缺失值
        df = df.dropna(subset=["compound__smiles", "protein_sequence"]).reset_index(drop=True)
        
        # 确保类型为字符串
        df["compound__smiles"] = df["compound__smiles"].astype(str)
        df["protein_sequence"] = df["protein_sequence"].astype(str)
        
        # 过滤掉空字符串和过短的序列
        df = df[
            (df["compound__smiles"].str.strip() != "") & 
            (df["protein_sequence"].str.strip() != "") &
            (df["protein_sequence"].str.len() >= 10)  # 至少10个氨基酸
        ].reset_index(drop=True)
        
        self.df = df
        print(f"Dataset size: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = str(row["compound__smiles"]).strip()
        protein_seq = str(row["protein_sequence"]).strip()
        
        if not smiles or not protein_seq:
            raise ValueError(f"Empty value at index {idx}")
        
        return {
            "smiles": smiles,
            "protein_sequence": protein_seq,
        }


def collate_fn(batch, mol_tokenizer, prot_tokenizer):
    smiles_list = []
    prot_list = []
    
    for b in batch:
        smiles = str(b["smiles"]).strip()
        prot = str(b["protein_sequence"]).strip()
        
        if not smiles or not prot:
            continue
        
        smiles_list.append(smiles)
        prot_list.append(prot)
    
    if len(smiles_list) == 0:
        raise ValueError("Batch is empty after filtering")
    
    mol_enc = mol_tokenizer(
        smiles_list,
        padding=True,
        truncation=True,
        max_length=cfg.max_len_mol,
        return_tensors="pt",
    )

    prot_enc = prot_tokenizer(
        prot_list,
        padding=True,
        truncation=True,
        max_length=cfg.max_len_prot,
        return_tensors="pt",
    )

    return mol_enc, prot_enc


# ===================== 改进的模型定义 =====================

class ChemBERTaEncoder(nn.Module):
    def __init__(self, model_name_or_path, projection_dim, freeze_backbone=True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = self.backbone.config.hidden_size
        
        # 改进的投影层：两层MLP + LayerNorm
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Linear(projection_dim * 2, projection_dim),
        )
        
        # 改进的初始化
        self._init_projection()
        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _init_projection(self):
        """改进的投影层初始化"""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

    def forward(self, **inputs):
        outputs = self.backbone(**inputs)
        
        if cfg.use_mean_pooling:
            # 平均池化：考虑attention mask
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                # [B, L, H] -> [B, H]
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = outputs.last_hidden_state.mean(dim=1)
        else:
            # CLS token
            pooled = outputs.last_hidden_state[:, 0]
        
        z = self.projection(pooled)
        z = nn.functional.normalize(z, p=2, dim=-1)
        return z


class SaProtEncoder(nn.Module):
    def __init__(self, model_name_or_path, projection_dim, freeze_backbone=True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = self.backbone.config.hidden_size
        
        # 改进的投影层：两层MLP + LayerNorm
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Linear(projection_dim * 2, projection_dim),
        )
        
        # 改进的初始化
        self._init_projection()
        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _init_projection(self):
        """改进的投影层初始化"""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

    def forward(self, **inputs):
        outputs = self.backbone(**inputs)
        
        if cfg.use_mean_pooling:
            # 平均池化：考虑attention mask
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = outputs.last_hidden_state.mean(dim=1)
        else:
            # CLS token
            pooled = outputs.last_hidden_state[:, 0]
        
        z = self.projection(pooled)
        z = nn.functional.normalize(z, p=2, dim=-1)
        return z


class DualEncoderModel(nn.Module):
    def __init__(self, chemberta_path, saprot_path, projection_dim):
        super().__init__()
        self.mol_encoder = ChemBERTaEncoder(
            chemberta_path, projection_dim, freeze_backbone=True
        )
        self.prot_encoder = SaProtEncoder(
            saprot_path, projection_dim, freeze_backbone=True
        )

    def forward(self, mol_inputs, prot_inputs):
        z_m = self.mol_encoder(**mol_inputs)
        z_p = self.prot_encoder(**prot_inputs)
        return z_m, z_p


# ===================== InfoNCE Loss =====================

def symmetric_info_nce_loss(z_m, z_p, temperature: float):
    batch_size = z_m.size(0)
    
    # 余弦相似度矩阵 [B, B]
    logits = torch.matmul(z_m, z_p.t()) / temperature
    
    labels = torch.arange(batch_size, device=logits.device)
    
    # 双向loss
    loss_m2p = nn.functional.cross_entropy(logits, labels)
    loss_p2m = nn.functional.cross_entropy(logits.t(), labels)
    
    loss = (loss_m2p + loss_p2m) / 2.0
    return loss


# ===================== 训练循环 =====================

def get_lr_scheduler(optimizer, num_epochs, warmup_epochs):
    """学习率调度器：预热 + 余弦退火"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1. 读数据
    print(f"Loading data from {cfg.data_path} ...")
    if cfg.data_path.endswith(".parquet"):
        df = pd.read_parquet(cfg.data_path)
    else:
        df = pd.read_csv(cfg.data_path)
    print("Data size:", len(df))

    # 2. 获取蛋白质序列（如果还没有）
    if 'protein_sequence' not in df.columns:
        print("⚠️  Warning: protein_sequence column not found!")
        print("Fetching protein sequences from UniProt...")
        df = load_or_fetch_sequences(df, cfg.protein_sequence_cache)
    else:
        print("✅ Using existing protein_sequence column")
        # 验证序列是否真的是序列而不是ID
        sample_seq = df['protein_sequence'].iloc[0] if len(df) > 0 else ""
        if len(str(sample_seq)) < 20:
            print(f"⚠️  Warning: Sequence too short ({len(str(sample_seq))} chars). Might be UniProt ID!")
            print(f"   Sample: {sample_seq}")
            print("   Fetching real sequences from UniProt...")
            df = load_or_fetch_sequences(df, cfg.protein_sequence_cache)
        else:
            print(f"✅ Sequences look valid (sample length: {len(str(sample_seq))})")
    
    # 验证序列数据
    print(f"\n验证序列数据:")
    print(f"  总样本数: {len(df)}")
    if 'protein_sequence' in df.columns:
        seq_lengths = df['protein_sequence'].astype(str).str.len()
        print(f"  序列长度统计: min={seq_lengths.min()}, max={seq_lengths.max()}, mean={seq_lengths.mean():.1f}")
        print(f"  前3个序列示例:")
        for i in range(min(3, len(df))):
            seq = str(df['protein_sequence'].iloc[i])
            print(f"    [{i}] {seq[:50]}... (长度: {len(seq)})")

    # 3. 创建数据集并划分
    full_dataset = DrugTargetDataset(df)
    print("Total positive pairs:", len(full_dataset))
    
    indices = list(range(len(full_dataset)))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.15, random_state=42, shuffle=True
    )
    
    train_df = full_dataset.df.iloc[train_indices].reset_index(drop=True)
    test_df = full_dataset.df.iloc[test_indices].reset_index(drop=True)
    
    train_dataset = DrugTargetDataset(train_df)
    test_dataset = DrugTargetDataset(test_df)
    
    print(f"Training pairs: {len(train_dataset)}")
    print(f"Test pairs: {len(test_dataset)}")

    # 4. 初始化 tokenizer & dataloader
    mol_tokenizer = AutoTokenizer.from_pretrained(cfg.chemberta_path)
    prot_tokenizer = AutoTokenizer.from_pretrained(cfg.saprot_path)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda batch: collate_fn(batch, mol_tokenizer, prot_tokenizer),
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda batch: collate_fn(batch, mol_tokenizer, prot_tokenizer),
    )

    # 5. 初始化模型
    model = DualEncoderModel(
        chemberta_path=cfg.chemberta_path,
        saprot_path=cfg.saprot_path,
        projection_dim=cfg.projection_dim,
    ).to(cfg.device)

    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数统计:")
    print(f"  可训练参数: {trainable_params:,} / {total_params:,}")
    
    # 检查投影层参数
    print(f"\n投影层参数检查:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")
    
    # 测试一个batch，检查embedding是否不同
    print(f"\n测试模型输出:")
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(train_dataloader))
        mol_inputs = {k: v.to(cfg.device) for k, v in test_batch[0].items()}
        prot_inputs = {k: v.to(cfg.device) for k, v in test_batch[1].items()}
        z_m, z_p = model(mol_inputs, prot_inputs)
        print(f"  分子embedding形状: {z_m.shape}")
        print(f"  蛋白质embedding形状: {z_p.shape}")
        print(f"  分子embedding前3个样本的L2范数: {torch.norm(z_m[:3], dim=1).tolist()}")
        print(f"  蛋白质embedding前3个样本的L2范数: {torch.norm(z_p[:3], dim=1).tolist()}")
        # 检查是否所有embedding都一样
        z_m_std = torch.std(z_m, dim=0).mean().item()
        z_p_std = torch.std(z_p, dim=0).mean().item()
        print(f"  分子embedding标准差: {z_m_std:.6f} (如果接近0说明所有样本输出相同!)")
        print(f"  蛋白质embedding标准差: {z_p_std:.6f} (如果接近0说明所有样本输出相同!)")
    model.train()

    # 6. 优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    
    scheduler = get_lr_scheduler(optimizer, cfg.num_epochs, cfg.warmup_epochs)

    # 7. 训练循环
    train_losses = []
    best_loss = float('inf')

    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        for step, (mol_inputs, prot_inputs) in enumerate(train_dataloader):
            mol_inputs = {k: v.to(cfg.device) for k, v in mol_inputs.items()}
            prot_inputs = {k: v.to(cfg.device) for k, v in prot_inputs.items()}

            z_m, z_p = model(mol_inputs, prot_inputs)
            loss = symmetric_info_nce_loss(z_m, z_p, cfg.temperature)

            optimizer.zero_grad()
            loss.backward()
            
            # 检查梯度（仅在第一个batch）
            if epoch == 0 and step == 0:
                print(f"\n梯度检查（第一个batch）:")
                total_grad_norm = 0.0
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_grad_norm += grad_norm ** 2
                        if grad_norm > 0:
                            print(f"  {name}: grad_norm={grad_norm:.6f}")
                total_grad_norm = total_grad_norm ** 0.5
                print(f"  总梯度范数: {total_grad_norm:.6f}")
                if total_grad_norm < 1e-6:
                    print(f"  ⚠️  警告: 梯度非常小，模型可能没有更新！")
            
            # 梯度裁剪
            if cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.gradient_clip
                )
            
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if (step + 1) % 10 == 0:
                avg_loss = running_loss / num_batches
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch [{epoch+1}/{cfg.num_epochs}] Step [{step+1}/{len(train_dataloader)}] "
                    f"Loss: {avg_loss:.4f} LR: {current_lr:.2e}"
                )

        # 更新学习率
        scheduler.step()

        # 计算该epoch的平均loss
        epoch_avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        train_losses.append(epoch_avg_loss)
        print(f"Epoch [{epoch+1}/{cfg.num_epochs}] Average Training Loss: {epoch_avg_loss:.4f}")

        # 保存最佳模型
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            best_ckpt_path = os.path.join(cfg.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_avg_loss,
                "config": cfg.__dict__,
            }, best_ckpt_path)
            print(f"Saved best model (loss: {best_loss:.4f}) to {best_ckpt_path}")

        # 定期保存checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(cfg.output_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg.__dict__,
            }, ckpt_path)

    # 8. 保存最终模型和绘制loss曲线
    final_dir = os.path.join(cfg.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "dual_encoder_state_dict.pt"))
    print(f"Training finished. Final weights saved to {final_dir}")
    
    # 绘制loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_plot_path = os.path.join(cfg.output_dir, "training_loss.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"Training loss curve saved to {loss_plot_path}")
    
    # 保存loss数据
    loss_data_path = os.path.join(cfg.output_dir, "training_loss.txt")
    with open(loss_data_path, 'w') as f:
        for epoch, loss in enumerate(train_losses, 1):
            f.write(f"Epoch {epoch}: {loss:.6f}\n")
    print(f"Training loss data saved to {loss_data_path}")


if __name__ == "__main__":
    main()

