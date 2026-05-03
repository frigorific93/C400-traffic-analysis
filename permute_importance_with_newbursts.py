#!/usr/bin/env python3
"""Compute permutation importance for the with_newbursts feature set.

Saves: baseline_results/perm_importance_with_newbursts.json and PNG.
"""
import os
import json
import math
from pprint import pformat

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
except Exception:
    lgb = None

from cv_baseline import DFPreprocessor, DEFAULT_FEATURES, HEAVY_TAIL_COLS


OUTDIR = 'baseline_results'
os.makedirs(OUTDIR, exist_ok=True)


def detect_burst_features(df):
    # heuristics: columns containing 'burst' or pattern '_t20' '_t50' '_t200' or 'burst_count' etc
    burst_cols = [c for c in df.columns if ('burst' in c or '_t20' in c or '_t50' in c or '_t200' in c)]
    return burst_cols


def load_data(csv='traffic_dataset.csv'):
    if not os.path.exists(csv):
        raise FileNotFoundError(csv)
    df = pd.read_csv(csv)
    return df


def main():
    df = load_data()
    # feature set: DEFAULT_FEATURES + any detected burst cols that exist
    base = [c for c in DEFAULT_FEATURES if c in df.columns]
    burst = [c for c in detect_burst_features(df) if c not in base]
    feats = base + burst
    print('Using', len(feats), 'features (base', len(base), '+ burst', len(burst), ')')

    # drop rows missing key things
    df = df.dropna(subset=feats + ['label', 'source_file'])
    X = df[feats]
    y = df['label'].astype(int)

    # split a holdout to compute importances deterministically
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pre = DFPreprocessor(feature_names=feats, heavy_tail=HEAVY_TAIL_COLS)
    scaler = StandardScaler()

    # build classifier
    if lgb is None:
        raise RuntimeError('lightgbm not installed in the environment')

    clf = lgb.LGBMClassifier(n_estimators=500, random_state=42, n_jobs=4)

    pipe = Pipeline([('pre', pre), ('scaler', scaler), ('clf', clf)])

    print('Fitting model for permutation importance...')
    pipe.fit(Xtr, ytr)

    # compute permutation importance on validation set
    print('Running permutation_importance (n_repeats=10)...')
    res = permutation_importance(pipe, Xval, yval, n_repeats=10, random_state=42, n_jobs=4, scoring='f1_macro')

    # summarise
    importances = []
    for i, name in enumerate(feats):
        imp = float(res.importances_mean[i])
        sd = float(res.importances_std[i])
        importances.append({'feature': name, 'mean_importance': imp, 'std': sd})

    importances = sorted(importances, key=lambda x: -x['mean_importance'])

    out_json = os.path.join(OUTDIR, 'perm_importance_with_newbursts.json')
    with open(out_json, 'w') as f:
        json.dump({'features': importances, 'n_features': len(feats)}, f, indent=2)

    # simple bar plot of top 20
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        top = importances[:20]
        names = [t['feature'] for t in top]
        vals = [t['mean_importance'] for t in top]
        errs = [t['std'] for t in top]
        plt.figure(figsize=(8,4))
        plt.barh(range(len(names))[::-1], vals, xerr=errs)
        plt.yticks(range(len(names))[::-1], names)
        plt.xlabel('Permutation importance (mean f1_macro drop)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, 'perm_importance_with_newbursts_top20.png'))
        plt.close()
    except Exception as e:
        print('Could not create plot:', e)

    print('Saved', out_json)


if __name__ == '__main__':
    main()
