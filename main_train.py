#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROI Transformer 年齢予測モデルの学習

目的：
    OpenBHBの解剖学的特徴量を用いて，
    Transformerによる年齢予測モデルを学習する．

入力：
    ・OpenBHBの学習用ROI特徴量
    ・正解年齢

処理：
    ・データをTrain / Validationに分割
    ・学習データから平均・標準偏差を計算
    ・ROI特徴量を正規化
    ・皮質のみ／皮質＋深部のモデルを選択
    ・Transformerを学習
    ・Validationデータで予測精度を評価

出力：
    ・学習済みモデル（best.pth）
    ・実験設定（config.json）
    ・学習・検証結果

備考：
    ・学習後にmain_test.pyによる評価を実行
"""

import os
import time
import sys
import random
import math
from tqdm import tqdm
import json


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from openbhb import OpenBHBDataset
from normalization import compute_mean_std, NormalizedDataset
from models import (
    ROITransformer,
    ROITransformerWithDeep,
)
from main_test import main as test_main
# from models import (
#     roi_transformer_tiny,
#     roi_transformer_small,
#     roi_transformer_base,
#     roi_transformer_large,
# )


# MODEL_REGISTRY = {
#     "roi_transformer_tiny": roi_transformer_tiny,
#     "roi_transformer_small": roi_transformer_small,
#     "roi_transformer_base": roi_transformer_base,
#     "roi_transformer_large": roi_transformer_large,
# }

#------------------------------
# Config
#------------------------------
class Config:
    # experiment settings
    EXPERIMENT_NAME = "exp0001_baseline_test"

    # dataset settings
    BASE_DIR = "/home/naoya/workspace/FreeSurfer/train"
    TSV_PATH = "/home/naoya/workspace/Data/OpenBHB/train/train_labels/official_site_class_labels.tsv"
    TRAIN_RATIO = 0.9

    # output settings
    OUT_ROOT = "/home/naoya/workspace/FreeSurfer_output"
    OUT_DIR = os.path.join(OUT_ROOT, EXPERIMENT_NAME)
    SAVE_FREQ = 10

    USE_DEEP = False

    NUM_DEEP_ROI = 45
    NUM_DEEP_FEATURES = 7

    # Model settings
    # MODEL = "roi_transformer_base"
    NUM_ROI = 68
    NUM_FEATURES = 9
    HIDDEN_SIZE = 256
    MLP_DIM = 1024
    NUM_LAYERS = 4
    NUM_HEADS = 8
    DROPOUT_RATE = 0.0

    # Training Settings
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 5e-2
    WARMUP_EPOCHS = 5
    MIN_LR_RATIO = 0.05

    # Dataloader Settings
    NUM_WORKERS = 8

    # Seed Settings
    SEED = 42
    #SEED = 3407
    

#------------------------------
# Utility
#------------------------------
def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#------------------------------
# LR Scheduler
#------------------------------
def build_scheduler(
    optimizer, total_steps, warmup_steps, min_lr_ratio,) -> torch.optim.lr_scheduler.LambdaLR:
    # Linear Warmup
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        # Cosine Decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def save_config(cfg):
    config_dict = {
        key: value
        for key, value in vars(cfg.__class__).items()
        if key.isupper()
    }

    config_path = os.path.join(cfg.OUT_DIR, "config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)

    print(f"Config saved: {config_path}")

#------------------------------
# Main
#------------------------------
def main():
    # Config
    cfg = Config()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    save_config(cfg)

    # set random seed
    set_random_seed(cfg.SEED)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # dataset
    dataset = OpenBHBDataset(cfg.BASE_DIR, cfg.TSV_PATH)

    # split train / validation
    n = len(dataset)
    n_train = int(n*cfg.TRAIN_RATIO)
    n_val = n - n_train
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.SEED)
    )
    print(f"Train data: {len(train_set)}")
    print(f"Validation data: {len(val_set)}")

    # Compute mean / std
    print("Compute mean / std")
    cortex_mean, cortex_std, deep_mean, deep_std = compute_mean_std(
        dataset,
        train_set.indices
    )

    # Normalization
    train_set = NormalizedDataset(train_set, cortex_mean, cortex_std, deep_mean, deep_std)
    val_set = NormalizedDataset(val_set, cortex_mean, cortex_std, deep_mean, deep_std)
    print(f"Cortex_Mean shape: {cortex_mean.shape}")
    print(f"Cortex_Std shape: {cortex_std.shape}")
    print(f"Deep_Mean shape: {deep_mean.shape}")
    print(f"Deep_Std shape: {deep_std.shape}")


    # DataLoader
    dl_train = DataLoader(
        train_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS
    )

    dl_val = DataLoader(
        val_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    # Model
    if cfg.USE_DEEP:
        model = ROITransformerWithDeep(
            num_roi=cfg.NUM_ROI,
            num_features=cfg.NUM_FEATURES,
            num_deep_roi=cfg.NUM_DEEP_ROI,
            num_deep_features=cfg.NUM_DEEP_FEATURES,
            hidden_size=cfg.HIDDEN_SIZE,
            mlp_dim=cfg.MLP_DIM,
            num_layers=cfg.NUM_LAYERS,
            num_heads=cfg.NUM_HEADS,
            dropout_rate=cfg.DROPOUT_RATE,
        )
    else:
        model = ROITransformer(
            num_roi=cfg.NUM_ROI,
            num_features=cfg.NUM_FEATURES,
            hidden_size=cfg.HIDDEN_SIZE,
            mlp_dim=cfg.MLP_DIM,
            num_layers=cfg.NUM_LAYERS,
            num_heads=cfg.NUM_HEADS,
            dropout_rate=cfg.DROPOUT_RATE,
        )
    model = model.to(device)

    # Loss
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    # Scheduler
    total_steps = cfg.EPOCHS * len(dl_train)
    warmup_steps = cfg.WARMUP_EPOCHS * len(dl_train)
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps, min_lr_ratio=cfg.MIN_LR_RATIO)

    #------------------------------
    # Training Loop
    #------------------------------
    print("Training start.")

    best_val_mae = float("inf")

    for epoch in range(1, cfg.EPOCHS + 1):
        time_start = time.time()

        #------------------------------
        # Train
        #------------------------------
        model.train()

        total_train_loss = 0.0

        train_bar = tqdm(
            dl_train,
            desc=f"Train [{epoch:04d}/{cfg.EPOCHS}]",
            leave=False,
        )

        for cortex_feature, deep_feature, age in train_bar:

            cortex_feature = cortex_feature.to(device)
            deep_feature = deep_feature.to(device)
            age = age.to(device)

            # Forward
            if cfg.USE_DEEP:
                pred_age = model(
                    cortex_feature,
                    deep_feature,
                )
            else:
                pred_age = model(cortex_feature)

            # Loss
            loss = criterion(pred_age, age)

            # Initialize Gradient
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Update Parameters
            optimizer.step()

            # Update Learning Rate
            scheduler.step()

            # Add Loss
            total_train_loss += loss.item()

            # Update tqdm
            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        # Train Loss
        train_loss = (total_train_loss / len(dl_train))

        #------------------------------
        # Validation
        #------------------------------
        model.eval()

        total_val_loss = 0.0
        total_absolute_error = 0.0
        total_val_samples = 0

        val_bar = tqdm(
            dl_val,
            desc=f"Val   [{epoch:04d}/{cfg.EPOCHS}]",
            leave=False,
        )

        with torch.no_grad():
            for cortex_feature, deep_feature, age in val_bar:
                # Move to Device
                cortex_feature = cortex_feature.to(device)
                deep_feature = deep_feature.to(device)
                age = age.to(device)

                # Forward
                if cfg.USE_DEEP:
                    pred_age = model(
                        cortex_feature,
                        deep_feature,
                    )
                else:
                    pred_age = model(cortex_feature)

                # Loss
                loss = criterion(pred_age, age)

                # Add Loss
                total_val_loss += loss.item()

                # Add Absolute Error
                absolute_error = torch.abs(pred_age - age)

                total_absolute_error += (absolute_error.sum().item())

                # Add Number of Samples
                total_val_samples += age.size(0)

                # Update tqdm
                val_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mae=f"{absolute_error.mean().item():.4f}",
                )

        # Validation Loss
        val_loss = (total_val_loss / len(dl_val))

        # Validation MAE
        val_mae = (total_absolute_error / total_val_samples)

        #------------------------------
        # Training Status
        #------------------------------
        elapsed = time.time() - time_start

        lr_now = scheduler.get_last_lr()[0]

        print(
            f"[{epoch:04d}/{cfg.EPOCHS}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_mae={val_mae:.4f} "
            f"lr={lr_now:.2e} "
            f"time={elapsed:.1f}s"
        )

        #------------------------------
        # Save Best Model
        #------------------------------
        if val_mae < best_val_mae:
            best_val_mae = val_mae

            best_model_path = os.path.join(cfg.OUT_DIR, "best.pth")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "use_deep": cfg.USE_DEEP,
                    "cortex_mean": cortex_mean,
                    "cortex_std": cortex_std,
                    "deep_mean": deep_mean,
                    "deep_std": deep_std,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                },
                best_model_path
            )

            print(
                f"Best model saved: "
                f"epoch={epoch}, "
                f"val_mae={val_mae:.4f}"
            )
            print(f"  → Saved best model (epoch={epoch}, val_loss={val_loss:.4f}, val_mae={val_mae:.4f})")

        #------------------------------
        # Save Checkpoint
        #------------------------------
        if epoch % cfg.SAVE_FREQ == 0:
            checkpoint_path = os.path.join(cfg.OUT_DIR, f"checkpoint_epoch_{epoch:04d}.pth")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "use_deep": cfg.USE_DEEP,
                    "cortex_mean": cortex_mean,
                    "cortex_std": cortex_std,
                    "deep_mean": deep_mean,
                    "deep_std": deep_std,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                },
                checkpoint_path
            )

            print(f"Checkpoint saved: {checkpoint_path}")

        print("Training finished.")

    print("Training complete finished.")

    #------------------------------
    # Test
    #------------------------------
    test_main(cfg.OUT_DIR, use_deep=cfg.USE_DEEP)

if __name__ == "__main__":
    main()