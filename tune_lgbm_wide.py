#!/usr/bin/env python3
"""Wide LightGBM tuning sweep for multiclass flows.

Runs a broader grid across key hyperparameters using StratifiedGroupKFold
and in-fold RandomOverSampler. Results are sorted by macro-F1 and saved to JSON.
"""
import argparse
import json
import os
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier

from cv_baseline import DFPreprocessor, HEAVY_TAIL_COLS, DEFAULT_FEATURES


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
    return [c for c in candidate if c in df.columns]


def evaluate_config(df, feats, cfg, n_splits=5, seed=42):
    Xdf = df[feats]
    y = df['label'].astype(int).values
    groups = df['source_file'].values

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pre = DFPreprocessor(feature_names=feats, heavy_tail=HEAVY_TAIL_COLS)

    accs, f1s = [], []
    for train_idx, test_idx in skf.split(Xdf, y, groups=groups):
        Xtr_df = Xdf.iloc[train_idx]
        Xte_df = Xdf.iloc[test_idx]
        ytr = y[train_idx]
        yte = y[test_idx]

        Xtr_arr = pre.transform(Xtr_df)
        Xte_arr = pre.transform(Xte_df)

        ros = RandomOverSampler(random_state=seed)
        X_res, y_res = ros.fit_resample(Xtr_arr, ytr)

        clf = LGBMClassifier(
            n_estimators=cfg['n_estimators'],
            learning_rate=cfg['learning_rate'],
            num_leaves=cfg['num_leaves'],
            max_depth=cfg['max_depth'],
            min_data_in_leaf=cfg['min_data_in_leaf'],
            feature_fraction=cfg['feature_fraction'],
            bagging_fraction=cfg['bagging_fraction'],
            bagging_freq=cfg['bagging_freq'],
            lambda_l1=cfg['lambda_l1'],
            lambda_l2=cfg['lambda_l2'],
            class_weight=cfg['class_weight'],
            random_state=seed,
        )
        clf.fit(X_res, y_res)
        preds = clf.predict(Xte_arr)

        accs.append(float(accuracy_score(yte, preds)))
        f1s.append(float(f1_score(yte, preds, average='macro')))

    return float(np.mean(accs)), float(np.mean(f1s))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='traffic_dataset.csv')
    parser.add_argument('--out', default='baseline_results/lgbm_tuning_wide.json')
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=['label', 'source_file'])
    feats = build_feature_list(df)

    grid = {
        'n_estimators': [200, 500, 800],
        'learning_rate': [0.03, 0.05, 0.1],
        'num_leaves': [31, 63, 127],
        'max_depth': [-1, 8, 12],
        'min_data_in_leaf': [20, 50, 100],
        'feature_fraction': [0.8, 0.9, 1.0],
        'bagging_fraction': [0.8, 0.9, 1.0],
        'bagging_freq': [1],
        'lambda_l1': [0.0, 0.1, 0.5],
        'lambda_l2': [0.0, 0.1, 0.5],
        'class_weight': [None, {0: 1.0, 1: 1.2, 2: 1.2}, {0: 1.0, 1: 1.4, 2: 1.4}],
    }

    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*grid.values())]
    print('Evaluating', len(combos), 'configs on', args.csv)

    results = []
    for i, cfg in enumerate(combos, 1):
        acc, f1 = evaluate_config(df, feats, cfg, n_splits=args.n_splits, seed=args.seed)
        results.append({'config': cfg, 'cv_mean_acc': acc, 'cv_mean_f1_macro': f1})
        print(f"[{i}/{len(combos)}] acc={acc:.4f} f1={f1:.4f} cfg={cfg}")

    results_sorted = sorted(results, key=lambda x: x['cv_mean_f1_macro'], reverse=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'csv': args.csv, 'results': results_sorted}, f, indent=2)

    best = results_sorted[0]
    print('\nBest config by macro-F1:')
    print(best)


if __name__ == '__main__':
    main()
