
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ADNI 疾患分類用データ読み込み

目的：
    FreeSurferで解析したADNIのROI特徴量を読み込む．

入力：
    ・lh.aparc.stats，rh.aparc.stats
    ・aseg.stats
    ・疾患ラベル（AD，MCI，CN）

処理：
    ・皮質68 ROI × 9特徴量を取得
    ・深部45 ROI × 7特徴量を取得
    ・解析に失敗した被験者を除外
    ・ディレクトリ名から疾患ラベルを取得

出力：
    ・皮質ROI特徴量
    ・深部ROI特徴量
    ・疾患ラベル
"""

import os
import time
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# 解析失敗して使用できないデータ
# ------------------------------
IGNORED_SUBJECT_ID = {
    "CN": (
        "subject000152",
        "subject000198",
        "subject000363",
        "subject000364",
        "subject000554",
    ),
    "MCI": (
        "subject000082",
    ),
    "AD": (
        "subject000198",
    ),
}

# ------------------------------
# Config
# ------------------------------
CLASS_TO_IDX = {
    "CN": 0,
    "MCI": 1,
    "AD": 2,
}


# ------------------------------
# load cortex stats
# ------------------------------
def load_cortex_stats(path):
    features = []

    with open(path) as f:
        for line in f:

            if line.startswith("#"):
                continue

            if not line.strip():
                continue

            values = line.split()

            if len(values) != 10:
                raise ValueError(
                    f"Unexpected cortex stats row: {line.strip()}, "
                    f"path={path}"
                )

            feature = [float(x) for x in values[1:]]
            features.append(feature)

    features = np.asarray(features, dtype=np.float32)

    if features.shape != (34, 9):
        raise ValueError(
            f"Unexpected cortex feature shape: {features.shape}, "
            f"path={path}"
        )

    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite cortex features: {path}")

    return features


# ------------------------------
# load deep stats
# ------------------------------
def load_deep_stats(path):
    features = []

    with open(path) as f:
        for line in f:

            if line.startswith("#"):
                continue

            if not line.strip():
                continue

            values = line.split()

            if len(values) < 10:
                raise ValueError(
                    f"Unexpected aseg.stats row: {line.strip()}, "
                    f"path={path}"
                )

            feature = [
                float(values[2]),  # NVoxels
                float(values[3]),  # Volume_mm3
                float(values[5]),  # normMean
                float(values[6]),  # normStdDev
                float(values[7]),  # normMin
                float(values[8]),  # normMax
                float(values[9]),  # normRange
            ]

            features.append(feature)

    features = np.asarray(features, dtype=np.float32)

    if features.shape != (45, 7):
        raise ValueError(
            f"Unexpected deep feature shape: {features.shape}, "
            f"path={path}"
        )

    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite deep features: {path}")

    return features


# ------------------------------
# Dataset
# ------------------------------
class ADNIDataset(Dataset):
    def __init__(self, base_dir, use_deep=True):
        self.base_dir = base_dir
        self.use_deep = use_deep
        self.samples = []
        self.excluded = []

        # 各疾患クラスを読み込む
        for class_name, label in CLASS_TO_IDX.items():

            class_dir = os.path.join(base_dir, class_name)

            if not os.path.isdir(class_dir):
                raise FileNotFoundError(
                    f"Class directory not found: {class_dir}"
                )

            subjects = sorted(os.listdir(class_dir))

            # 解析失敗した被験者を除外
            subjects = [
                s for s in subjects
                if s not in IGNORED_SUBJECT_ID[class_name]
            ]

            for subject in subjects:

                subject_dir = os.path.join(class_dir, subject)

                if not os.path.isdir(subject_dir):
                    continue

                stats_dir = os.path.join(subject_dir, "stats")

                lh_path = os.path.join(
                    stats_dir, "lh.aparc.stats"
                )
                rh_path = os.path.join(
                    stats_dir, "rh.aparc.stats"
                )
                aseg_path = os.path.join(
                    stats_dir, "aseg.stats"
                )

                # 必要なファイルの存在確認
                required_paths = [lh_path, rh_path]

                if self.use_deep:
                    required_paths.append(aseg_path)

                if not all(os.path.isfile(p) for p in required_paths):
                    self.excluded.append(
                        (class_name, subject, "Missing stats")
                    )
                    continue

                # # 特徴量の形状・値を確認
                # try:
                #     load_cortex_stats(lh_path)
                #     load_cortex_stats(rh_path)
                #     load_deep_stats(aseg_path)

                # except (ValueError, OSError) as e:
                #     self.excluded.append(
                #         (class_name, subject, str(e))
                #     )
                #     continue

                self.samples.append(
                    (subject_dir, label)
                )

        # データ数を表示
        print("Dataset Summary")
        print(f"Total samples: {len(self.samples)}")

        counts = Counter(
            label for _, label in self.samples
        )

        for class_name, label in CLASS_TO_IDX.items():
            print(
                f"{class_name}: {counts[label]}"
            )

        print(f"Excluded samples: {len(self.excluded)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        subject_dir, label = self.samples[index]

        stats_dir = os.path.join(subject_dir, "stats")

        lh_path = os.path.join(stats_dir, "lh.aparc.stats")
        rh_path = os.path.join(stats_dir, "rh.aparc.stats")

        # 皮質ROI
        lh_feature = load_cortex_stats(lh_path)
        rh_feature = load_cortex_stats(rh_path)

        cortex_feature = np.concatenate(
            [lh_feature, rh_feature], axis=0
        )

        cortex_feature = torch.from_numpy(cortex_feature).float()
        label = torch.tensor(label, dtype=torch.long)

        # 皮質ROIのみを使用
        if not self.use_deep:
            return cortex_feature, label

        # 脳深部ROIも使用
        aseg_path = os.path.join(stats_dir, "aseg.stats")
        deep_feature = load_deep_stats(aseg_path)
        deep_feature = torch.from_numpy(deep_feature).float()

        return cortex_feature, deep_feature, label



# ------------------------------
# Debug
# ------------------------------
if __name__ == "__main__":

    base_dir = "/home/naoya/workspace/Freesurfer_adni"

    # True：皮質ROI + 深部ROI
    # False：皮質ROIのみ
    use_deep = False

    dataset = ADNIDataset(
        base_dir,
        use_deep=use_deep
    )

    if len(dataset) == 0:
        raise RuntimeError("No valid samples found")

    # ------------------------------
    # Dataset Debug
    # ------------------------------
    print("\nDataset Debug")

    time_s = time.time()

    if use_deep:
        cortex_feature, deep_feature, label = dataset[0]
    else:
        cortex_feature, label = dataset[0]

    time_e = time.time()

    print("Cortex:", cortex_feature.shape)

    if use_deep:
        print("Deep:", deep_feature.shape)

    print("Label:", label)
    print(f"Load time: {time_e - time_s:.4f} s")

    # ------------------------------
    # DataLoader Debug
    # ------------------------------
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    print("\nDataLoader Debug")

    time_s = time.time()

    for batch in dataloader:

        if use_deep:
            cortex_feature, deep_feature, label = batch
        else:
            cortex_feature, label = batch

        print("Cortex:", cortex_feature.shape)

        if use_deep:
            print("Deep:", deep_feature.shape)

        print("Label:", label)
        break

    time_e = time.time()

    print(f"Batch load time: {time_e - time_s:.4f} s")

    # ------------------------------
    # Excluded Subjects
    # ------------------------------
    if dataset.excluded:
        print("\nExcluded Subjects")

        for class_name, subject, reason in dataset.excluded:
            print(f"{class_name}/{subject}: {reason}")

