#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROI Transformer 年齢予測モデルの定義

目的：
    FreeSurferから得られる解剖学的特徴量を入力とし，
    Transformer Encoderによって脳年齢を予測する．

モデル：
    1．ROITransformer
       ・皮質68 ROI × 9特徴量
       ・CLSトークンを使用して年齢を予測

    2．ROITransformerWithDeep
       ・皮質68 ROI × 9特徴量
       ・深部45 ROI × 7特徴量
       ・皮質・深部の特徴量を統合して年齢を予測

処理：
    ・ROI特徴量をEmbedding
    ・CLSトークンと位置Embeddingを追加
    ・Transformer Encoderで特徴を抽出
    ・CLSトークンから年齢を回帰

出力：
    ・予測年齢
"""

import os
import time
import numpy as np

import torch
import torch.nn as nn

class ROITransformer(nn.Module):
    def __init__(
        self,
        num_roi: int = 68,
        num_features: int = 9,
        num_classes: int = 3,
        hidden_size: int = 256,
        mlp_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
    ):

        super().__init__()

        # エラー処理
        if not (0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate should be between 0 and 1, got {dropout_rate}.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_roi = num_roi
        self.num_features = num_features
        self.hidden_size = hidden_size
        
        ### Encoderへの入力処理 ------------------------------------------
        # linear embedding
        # (B, 68, 9) -> (B, 68, hidden_size)
        self.roi_embedding = nn.Linear(num_features, hidden_size)

        # CLSトークンの作成
        # (1, 1, hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Positional Embedding
        # cls + 68 ROI = 69トークン分のEmbeddingを作成
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_roi + 1, hidden_size))

        # Transformer Encoder Layer （1層分）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout= dropout_rate,
            batch_first=True
        )

        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Layer Normalization
        self.norm = nn.LayerNorm(hidden_size)

        # 疾患分類用 MLP Head
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
    
    def forward(self, x, debug=False):
        """
        Args:
            x: ROI特徴量
                shape = (B, 68, 9)

        Returns:
            logits: 各疾患クラスの予測スコア
                shape = (B, num_classes)
        """
        if debug:
            print("input:", x.shape)

        # ROI特徴をLinear Embedding
        x = self.roi_embedding(x)
        if debug:
            print("after roi_embedding:", x.shape)
            print("roi_embedding sample:")
            print(x[:, 0, :5])

        # clsトークンをbatch size分拡張
        # (1, 1, hidden_size) -> (B, 1, hidden_size)
        cls_tokens = self.cls_token.expand(
            x.shape[0],
            -1,
            -1
        )
        if debug:
            print("cls_tokens:", cls_tokens.shape)

        # clsトークンをROIトークンの先頭に追加：トークン方向に連結
        # (B, 68, hidden_size) -> (B, 69, hidden_size)
        x = torch.cat(
            (cls_tokens, x),
            dim=1
        )
        if debug:
            print("after cat cls:", x.shape)

        # Positional Embeddingを追加
        # (B, 69, hidden_size)
        x = x + self.pos_embedding
        if debug:
            print("after pos_embedding:", x.shape)

        # Transformer Encoder
        # (B, 69, hidden_size) -> (B, 69, hidden_size)
        x = self.encoder(x)
        if debug:
            print("after encoder:", x.shape)

        # Layer Normalization
        x = self.norm(x)
        if debug:
            print("after norm:", x.shape)

        # clsトークンの最終出力を取得
        # (B, 69, hidden_size) -> (B, hidden_size)
        cls_last_token = x[:, 0]
        if debug:
            print("cls_last_token:", cls_last_token.shape)
            print("cls_last_token first 5 dims:")
            print(cls_last_token[:, :5])

        # CLSトークンから疾患クラスを予測
        # (B, hidden_size) -> (B, num_classes)
        logits = self.mlp_head(cls_last_token)

        return logits

class ROITransformerWithDeep(ROITransformer):
    def __init__(
        self,
        num_roi=68,
        num_features=9,
        num_deep_roi=45,
        num_deep_features=7,
        num_classes=3,
        hidden_size=256,
        mlp_dim=1024,
        num_layers=4,
        num_heads=8,
        dropout_rate=0.0,
    ):
        super().__init__(
            num_roi=num_roi,
            num_features=num_features,
            num_classes=num_classes,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

        # 深部構造Embedding
        self.deep_embedding = nn.Linear(
            num_deep_features,
            hidden_size,
        )

        # Positional Embeddingを114トークン用へ変更
        self.pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                1 + num_roi + num_deep_roi,
                hidden_size,
            )
        )

    def forward(self, cortex, deep, debug=False):
        if debug:
            print("cortex:", cortex.shape)
            print("deep:", deep.shape)
        # ROI Embedding
        cortex = self.roi_embedding(cortex)

        # Deep Embedding
        deep = self.deep_embedding(deep)

        # Tokenを連結
        x = torch.cat(
            [cortex, deep],
            dim=1,
        )

        # CLS追加
        cls_tokens = self.cls_token.expand(
            x.shape[0],
            -1,
            -1,
        )

        x = torch.cat(
            [cls_tokens, x],
            dim=1,
        )

        # Position Embedding
        x = x + self.pos_embedding

        # Encoder
        x = self.encoder(x)

        # LayerNorm
        x = self.norm(x)

        # CLS
        cls = x[:, 0]

        # 疾患分類
        logits = self.mlp_head(cls)

        return logits

if __name__ == "__main__":
    # モデルを作成
    model = ROITransformer(
        num_roi=68,
        num_features=9,
        num_classes=3,
        hidden_size=256,
        mlp_dim=1024,
        num_layers=4,
        num_heads=8,
        dropout_rate=0.0,
    )

    # Debug用のダミーデータ
    # batch_size=4
    # 68 ROI
    # 9特徴量
    x = torch.randn(4, 68, 9)

    # モデルへ入力
    logits= model(x, True)

    # Debug
    print("Model Debug")
    print("Input shape:")
    print(x.shape)

    print("Output shape:")
    print(logits.shape)

    print("logits:")
    print(logits)

    model = ROITransformerWithDeep(
        num_roi=68,
        num_features=9,
        num_deep_roi=45,
        num_deep_features=7,
        hidden_size=256,
        mlp_dim=1024,
        num_layers=4,
        num_heads=8,
        dropout_rate=0.0,
    )

    cortex = torch.randn(4, 68, 9)
    deep = torch.randn(4, 45, 7)

    logits = model(cortex, deep, True)

    print("\nROITransformerWithDeep Debug")
    print("Cortex:", cortex.shape)
    print("Deep:", deep.shape)
    print("Logits:", logits.shape)
    print("Predicted class:", logits.argmax(dim=1))


# ==========================================================
# ROI Transformer Variants
# ==========================================================

def roi_transformer_tiny(
    num_roi: int = 68,
    num_features: int = 9,
    **kwargs,
) -> ROITransformer:
    """ROI Transformer Tiny"""

    return ROITransformer(
        num_roi=num_roi,
        num_features=num_features,
        hidden_size=128,
        mlp_dim=128 * 4,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.0,
        **kwargs,
    )


def roi_transformer_small(
    num_roi: int = 68,
    num_features: int = 9,
    **kwargs,
) -> ROITransformer:
    """ROI Transformer Small"""

    return ROITransformer(
        num_roi=num_roi,
        num_features=num_features,
        hidden_size=192,
        mlp_dim=192 * 4,
        num_layers=4,
        num_heads=6,
        dropout_rate=0.0,
        **kwargs,
    )


def roi_transformer_base(
    num_roi: int = 68,
    num_features: int = 9,
    **kwargs,
) -> ROITransformer:
    """ROI Transformer Base"""

    return ROITransformer(
        num_roi=num_roi,
        num_features=num_features,
        hidden_size=256,
        mlp_dim=256 * 4,
        num_layers=6,
        num_heads=8,
        dropout_rate=0.0,
        **kwargs,
    )


def roi_transformer_large(
    num_roi: int = 68,
    num_features: int = 9,
    **kwargs,
) -> ROITransformer:
    """ROI Transformer Large"""

    return ROITransformer(
        num_roi=num_roi,
        num_features=num_features,
        hidden_size=512,
        mlp_dim=512 * 4,
        num_layers=8,
        num_heads=16,
        dropout_rate=0.0,
        **kwargs,
    )