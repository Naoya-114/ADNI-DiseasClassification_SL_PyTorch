
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROI特徴量の正規化

目的：
    ROIごと・特徴量ごとのスケールの違いを補正する．

入力：
    ・皮質ROI特徴量（68 ROI × 9特徴量）
    ・深部ROI特徴量（45 ROI × 7特徴量）
    ・学習用データのインデックス

処理：
    ・学習用データから平均・標準偏差を計算
    ・皮質ROIと深部ROIをそれぞれ正規化
    ・検証・評価時にも学習用データの統計量を使用
    ・皮質ROIのみ，皮質＋深部ROIの両方に対応

出力：
    ・各ROI特徴量の平均・標準偏差
    ・正規化済みROI特徴量
    ・疾患ラベル
"""

import torch
from torch.utils.data import Dataset


# ------------------------------
# Compute mean and std
# ------------------------------
def compute_mean_std(dataset, indices, use_deep=True):
    cortex_list = []
    deep_list = []

    # Trainデータの特徴量を取得
    for idx in indices:

        if use_deep:
            cortex_features, deep_features, _ = dataset[idx]
            deep_list.append(deep_features)
        else:
            cortex_features, _ = dataset[idx]

        cortex_list.append(cortex_features)

    # 皮質ROI
    cortex_features = torch.stack(cortex_list, dim=0)

    cortex_mean = cortex_features.mean(dim=0)
    cortex_std = cortex_features.std(dim=0)

    # 深部ROIを使用しない場合
    if not use_deep:
        return cortex_mean, cortex_std, None, None

    # 深部ROI
    deep_features = torch.stack(deep_list, dim=0)

    deep_mean = deep_features.mean(dim=0)
    deep_std = deep_features.std(dim=0)

    return cortex_mean, cortex_std, deep_mean, deep_std


# ------------------------------
# Normalized Dataset
# ------------------------------
class NormalizedDataset(Dataset):
    def __init__(
        self,
        dataset,
        cortex_mean,
        cortex_std,
        deep_mean=None,
        deep_std=None,
        use_deep=True,
    ):
        self.dataset = dataset
        self.cortex_mean = cortex_mean
        self.cortex_std = cortex_std
        self.deep_mean = deep_mean
        self.deep_std = deep_std
        self.use_deep = use_deep

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        if self.use_deep:
            cortex_feature, deep_feature, label = self.dataset[index]
        else:
            cortex_feature, label = self.dataset[index]

        # 皮質ROIの正規化
        cortex_feature = (
            cortex_feature - self.cortex_mean
        ) / (self.cortex_std + 1e-8)

        # 皮質ROIのみを使用
        if not self.use_deep:
            return cortex_feature, label

        # 深部ROIの正規化
        deep_feature = (
            deep_feature - self.deep_mean
        ) / (self.deep_std + 1e-8)

        return cortex_feature, deep_feature, label
