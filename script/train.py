"""
train.py — 完整训练流程
============================================================
最终配置 cdim192_ref:
  模型: 单尺度 CNN k=21, cnn_dim=192, CosineClassifier(temp=16)
  训练: CE + sqrt_inv weights + label_smoothing=0.1
        WeightedRandomSampler, AdamW lr=3e-4 wd=5e-3
        dropout=0.6, CosineAnnealing, patience=15

结果:
  CV:   ACC=0.8993, F1=0.8795
  Test: ACC=0.9392, F1=0.9042

用法:
  python train.py --device cuda:0
  python train.py --device cuda:0 --seed 123
"""

import argparse
import os
import json
import logging
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from tqdm import tqdm

from dataset import ProteinDataset
from model import SecretionModel, count_parameters

logger = logging.getLogger("train")


# ============================================================
# 配置
# ============================================================
CONFIG = {
    # 数据路径
    'dict_path':    '/work/data1/liutianyuan/wls/SPAN/data/dictTrain2918.pkl',
    'emb_dir':      '/work/data1/liutianyuan/wls/SPAN/protT5/train',
    'dict_test':    '/work/data1/liutianyuan/wls/SPAN/data/dictTest260.pkl',
    'emb_test':     '/work/data1/liutianyuan/wls/SPAN/protT5/test',

    # 模型
    'd_emb':              1024,
    'window_size':        100,
    'cnn_dim':            192,
    'kernel_size':        21,
    'hidden_global':      256,
    'hidden_fusion':      256,
    'dropout':            0.6,
    'input_dropout':      0.1,
    'num_classes':        6,
    'cosine_temperature': 16.0,

    # 训练
    'label_smoothing':    0.1,
    'lr':                 3e-4,
    'weight_decay':       5e-3,
    'epochs':             80,
    'batch_size':         64,
    'num_workers':        4,
    'n_folds':            5,
    'patience':           15,
    'seed':               42,
}


# ============================================================
# 工具
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output_dir):
    if logger.handlers:
        logger.handlers.clear()
    log_path = output_dir / "train.log"
    logger.setLevel(logging.INFO)
    logger.propagate = False
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(sh)
    logger.addHandler(fh)


def evaluate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    pr = precision_score(y_true, y_pred, average='macro', zero_division=0)
    sn = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    auprc = average_precision_score(y_true, y_prob, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = [], []
    for i in range(len(cm)):
        tni = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fpi = np.sum(np.delete(cm[:, i], i))
        tn.append(tni)
        fp.append(fpi)
    sp = np.mean(np.array(tn) / (np.array(tn) + np.array(fp) + 1e-12))
    return acc, sn, sp, pr, f1, mcc, auroc, auprc


def compute_class_weights(labels, device):
    counts = Counter(labels.tolist() if isinstance(labels, np.ndarray) else labels)
    num_classes = len(counts)
    total = sum(counts.values())
    n = torch.tensor([counts[i] for i in range(num_classes)], dtype=torch.float32)
    return torch.sqrt(total / (num_classes * n)).to(device)


def make_sampler(dataset, indices):
    labels = [dataset.data_dict[dataset.data_list[i]][-1] for i in indices]
    counts = Counter(labels)
    cw = {cls: 1.0 / cnt for cls, cnt in counts.items()}
    sw = [cw[l] for l in labels]
    return WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)


def build_model(cfg):
    return SecretionModel(
        d_emb=cfg['d_emb'],
        cnn_dim=cfg['cnn_dim'],
        kernel_size=cfg['kernel_size'],
        hidden_global=cfg['hidden_global'],
        hidden_fusion=cfg['hidden_fusion'],
        dropout=cfg['dropout'],
        input_dropout=cfg['input_dropout'],
        num_classes=cfg['num_classes'],
        cosine_temperature=cfg['cosine_temperature'],
    )


# ============================================================
# 评估
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for gx, ns, cs, lb in loader:
        gx, ns, cs, lb = gx.to(device), ns.to(device), cs.to(device), lb.to(device)
        logits = model(gx, ns, cs)
        probs = torch.softmax(logits, dim=-1)
        y_true.extend(lb.cpu().numpy())
        y_pred.extend(probs.argmax(dim=-1).cpu().numpy())
        y_prob.extend(probs.cpu().numpy())
    return evaluate_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))


# ============================================================
# 训练一个 fold
# ============================================================
def train_fold(model, train_loader, val_loader, cfg, device, fold, save_dir):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    labels_train = []
    for batch in train_loader:
        labels_train.extend(batch[-1].numpy())
    class_weights = compute_class_weights(np.array(labels_train), device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg['label_smoothing'])

    best_f1, best_metrics, best_epoch, pat_cnt = 0.0, None, 0, 0
    best_path = save_dir / f"best_model_fold_{fold}.pt"

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total_loss = 0.0
        for gx, ns, cs, lb in tqdm(train_loader, desc=f"F{fold} Ep{epoch}", leave=False):
            gx, ns, cs, lb = gx.to(device), ns.to(device), cs.to(device), lb.to(device)
            logits = model(gx, ns, cs)
            loss = criterion(logits, lb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        v_acc, _, _, _, v_f1, v_mcc, _, _ = val_metrics

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"  F{fold} Ep{epoch:02d}: Loss={total_loss/len(train_loader):.4f} "
                        f"ACC={v_acc:.4f} F1={v_f1:.4f} MCC={v_mcc:.4f}")

        if v_f1 > best_f1:
            best_f1, best_metrics, best_epoch, pat_cnt = v_f1, val_metrics, epoch, 0
            torch.save(model.state_dict(), best_path)
        else:
            pat_cnt += 1
            if pat_cnt >= cfg['patience']:
                logger.info(f"  F{fold} early stop ep{epoch} (best ep{best_epoch})")
                break

    logger.info(f"  F{fold} ✅ best@ep{best_epoch}: ACC={best_metrics[0]:.4f} F1={best_metrics[4]:.4f}")
    return best_metrics


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./output')
    args = parser.parse_args()

    cfg = {**CONFIG, 'seed': args.seed}
    set_seed(cfg['seed'])
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    logger.info("=" * 60)
    logger.info("Secretion Effector Prediction — Final Model")
    logger.info(f"  k=21, cnn_dim=192, CosineClassifier, WeightedSampler")
    logger.info(f"  dropout=0.6, wd=5e-3, seed={cfg['seed']}")
    logger.info("=" * 60)

    # 数据
    dataset = ProteinDataset(cfg['dict_path'], cfg['emb_dir'], window_size=cfg['window_size'])
    labels = np.array([dataset.data_dict[name][-1] for name in dataset.data_list])
    logger.info(f"Train: {len(labels)} samples, {Counter(sorted(labels.tolist()))}")

    model_tmp = build_model(cfg)
    logger.info(f"Parameters: {count_parameters(model_tmp):,}")
    del model_tmp

    # 5-fold CV
    skf = StratifiedKFold(n_splits=cfg['n_folds'], shuffle=True, random_state=cfg['seed'])
    all_fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        logger.info(f"\n--- Fold {fold}/{cfg['n_folds']} ---")
        set_seed(cfg['seed'])

        sampler = make_sampler(dataset, train_idx)
        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=cfg['batch_size'],
            shuffle=False, sampler=sampler,
            num_workers=cfg['num_workers'], pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx), batch_size=cfg['batch_size'],
            shuffle=False, num_workers=cfg['num_workers'], pin_memory=True,
        )

        model = build_model(cfg).to(device)
        metrics = train_fold(model, train_loader, val_loader, cfg, device, fold, output_dir)
        all_fold_metrics.append(metrics)

    # CV 汇总
    arr = np.array(all_fold_metrics)
    mean_cv, std_cv = arr.mean(0), arr.std(0)
    names = ["ACC", "SN", "SP", "PR", "F1", "MCC", "AUROC", "AUPRC"]

    logger.info(f"\n{'='*50}")
    logger.info("5-Fold Cross-Validation")
    logger.info(f"{'='*50}")
    cv_results = {}
    for n, m, s in zip(names, mean_cv, std_cv):
        logger.info(f"  {n}: {m:.4f} ± {s:.4f}")
        cv_results[n] = float(m)

    # 独立测试集
    logger.info(f"\n{'='*50}")
    logger.info("Independent Test Set")
    logger.info(f"{'='*50}")
    test_ds = ProteinDataset(cfg['dict_test'], cfg['emb_test'], window_size=cfg['window_size'])
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False,
                             num_workers=cfg['num_workers'], pin_memory=True)
    logger.info(f"Test: {len(test_ds)} samples")

    test_all = []
    for p in sorted(output_dir.glob("best_model_fold_*.pt")):
        m = build_model(cfg).to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        metrics = evaluate(m, test_loader, device)
        acc, sn, sp, pr, f1, mcc, auroc, auprc = metrics
        logger.info(f"  {p.name}: ACC={acc:.4f} F1={f1:.4f} MCC={mcc:.4f}")
        test_all.append(metrics)

    test_arr = np.array(test_all)
    mean_te, std_te = test_arr.mean(0), test_arr.std(0)
    logger.info(f"\n  Mean:")
    test_results = {}
    for n, m, s in zip(names, mean_te, std_te):
        logger.info(f"  {n}: {m:.4f} ± {s:.4f}")
        test_results[n] = float(m)

    # 保存
    summary = {'cv': cv_results, 'test': test_results, 'config': cfg}
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info("FINAL SUMMARY")
    logger.info(f"  CV:   ACC={cv_results['ACC']:.4f}  F1={cv_results['F1']:.4f}")
    logger.info(f"  Test: ACC={test_results['ACC']:.4f}  F1={test_results['F1']:.4f}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()