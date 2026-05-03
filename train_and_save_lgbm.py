#!/usr/bin/env python3
"""Train a LightGBM model on the full dataset and save a pipeline for inference.

This script mirrors the feature selection and preprocessing used in `eval_new_features.py`.
It will:
 - load `traffic_dataset.csv`
 - build the same feature list as the eval script
 - run a quick StratifiedGroupKFold CV (5 folds) to report cv mean accuracy and macro F1
 - fit a final pipeline on the full dataset
 - save the pipeline to `baseline_results/lgbm_newfeatures/lgbm_inference.joblib`
 - save a small JSON with CV metrics and model path

If lightgbm is not installed, the script will exit with a clear message.
"""
import os
import json
import joblib
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# import DFPreprocessor, DEFAULT_FEATURES, HEAVY_TAIL_COLS from cv_baseline
from cv_baseline import DFPreprocessor, DEFAULT_FEATURES, HEAVY_TAIL_COLS

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


def build_feature_list(df):
    candidate = list(DEFAULT_FEATURES)
    extras = [
        'first4_up_mean','first4_up_std','first4_down_mean','first4_down_std',
        'pkt_bin_p0','pkt_bin_p1','pkt_bin_p2','pkt_bin_p3','pkt_size_entropy',
        'pkt_bin_up_p0','pkt_bin_up_p1','pkt_bin_up_p2','pkt_bin_up_p3',
        'pkt_bin_down_p0','pkt_bin_down_p1','pkt_bin_down_p2','pkt_bin_down_p3',
        'pkt_size_entropy_up','pkt_size_entropy_down',
        'iat_up_median','iat_up_q10','iat_up_q90','iat_up_cv','iat_up_skew',
        'iat_down_median','iat_down_q10','iat_down_q90','iat_down_cv','iat_down_skew',
        'burst_iai_cv'
    ]
    for e in extras:
        if e in df.columns and e not in candidate:
            candidate.append(e)
    feats = [c for c in candidate if c in df.columns]
    return feats


def main(csv='traffic_dataset.csv', out_dir='baseline_results/lgbm_newfeatures', n_splits=5, seed=42):
    if LGBMClassifier is None:
        print('lightgbm not installed in this environment. Install with `pip install lightgbm` and retry.')
        return 1

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv):
        print('Dataset CSV not found:', csv)
        return 1

    df = pd.read_csv(csv)
    df = df.dropna(subset=['label','source_file'])

    feats = build_feature_list(df)
    if len(feats) == 0:
        print('No features detected in CSV. Check that feature extraction ran successfully.')
        return 1

    Xdf = df[feats]
    y = df['label'].astype(int).values
    groups = df['source_file'].values

    print('Using features (%d):' % len(feats), feats)

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_metrics = []
    pre = DFPreprocessor(feature_names=feats, heavy_tail=HEAVY_TAIL_COLS)

    for fold, (train_idx, test_idx) in enumerate(skf.split(Xdf, y, groups=groups)):
        Xtr = Xdf.iloc[train_idx]
        Xte = Xdf.iloc[test_idx]
        ytr = y[train_idx]
        yte = y[test_idx]

        Xtr_arr = pre.transform(Xtr)
        Xte_arr = pre.transform(Xte)

        clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=seed)
        clf.fit(Xtr_arr, ytr)

        preds = clf.predict(Xte_arr)
        acc = float(accuracy_score(yte, preds))
        f1 = float(f1_score(yte, preds, average='macro'))
        fold_metrics.append({'fold': int(fold), 'acc': acc, 'f1_macro': f1, 'n_test': int(len(yte))})
        print(f'fold {fold}: acc={acc:.4f} f1_macro={f1:.4f} (n_test={len(yte)})')

    accs = [m['acc'] for m in fold_metrics]
    f1s = [m['f1_macro'] for m in fold_metrics]
    summary = {
        'cv_mean_acc': float(np.mean(accs)),
        'cv_std_acc': float(np.std(accs)),
        'cv_mean_f1_macro': float(np.mean(f1s)),
        'cv_std_f1_macro': float(np.std(f1s)),
        'folds': fold_metrics
    }

    print('\nCV summary: mean acc %.4f ± %.4f, mean f1 %.4f ± %.4f' % (summary['cv_mean_acc'], summary['cv_std_acc'], summary['cv_mean_f1_macro'], summary['cv_std_f1_macro']))

    # fit final pipeline on entire dataset
    scaler = StandardScaler()
    clf_final = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=seed)

    pipeline = Pipeline([
        ('pre', DFPreprocessor(feature_names=feats, heavy_tail=HEAVY_TAIL_COLS)),
        ('scaler', scaler),
        ('clf', clf_final)
    ])

    print('Fitting pipeline on full dataset...')
    X_full = df[feats]
    y_full = df['label'].astype(int)
    pipeline.fit(X_full, y_full)

    model_path = os.path.join(out_dir, 'lgbm_inference.joblib')
    joblib.dump(pipeline, model_path)

    meta = {'model_path': model_path, 'features': feats, 'cv': summary}
    with open(os.path.join(out_dir, 'lgbm_inference_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print('Saved inference pipeline to', model_path)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='traffic_dataset.csv')
    parser.add_argument('--out', default='baseline_results/lgbm_newfeatures')
    parser.add_argument('--n-splits', type=int, default=5)
    args = parser.parse_args()
    raise SystemExit(main(csv=args.csv, out_dir=args.out, n_splits=args.n_splits))
