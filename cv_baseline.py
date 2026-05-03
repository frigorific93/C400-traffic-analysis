#!/usr/bin/env python3
"""Run GroupKFold CV with a small RandomForest grid search.

Saves: best model (joblib), grid search summary, per-fold metrics (json).
"""
import argparse
import json
import os
from pprint import pformat

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


HEAVY_TAIL_COLS = [
    'byte_count','upload_bytes','download_bytes','peak_rate_1s','peak_download_1s','peak_upload_1s',
    'avg_burst_bytes','max_burst_bytes','pkt_size_mean'
]

DEFAULT_FEATURES = [
    'packet_count','byte_count','upload_bytes','download_bytes','upload_packets','download_packets',
    'byte_ratio_up_down','packet_ratio_up_down','direction_imbalance',
    'flow_duration','iat_mean','iat_std','burst_count','avg_burst_bytes','peak_rate_1s',
    'peak_download_1s','peak_upload_1s','pkt_size_mean','pkt_size_std','pkt_q50'
]


class DFPreprocessor(BaseEstimator, TransformerMixin):
    """Transformer that applies log1p to selected heavy-tail cols and returns numpy array for features order."""
    def __init__(self, feature_names, heavy_tail=None):
        # IMPORTANT: do NOT modify constructor parameters here (store them as-is)
        # so sklearn.clone can recreate the estimator. Any conversions should
        # be done during transform.
        self.feature_names = feature_names
        self.heavy_tail = heavy_tail

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X may be a DataFrame or array; expect DataFrame
        if not hasattr(X, 'loc'):
            # assume it's already numeric array in correct order
            return X
        # Work from local copies and convert types here to avoid modifying
        # the original parameter objects stored on self.
        feature_list = list(self.feature_names)
        heavy_set = set(self.heavy_tail or [])
        Xsub = X[feature_list].copy()
        for c in feature_list:
            if c in heavy_set:
                # guard against negatives or NaN
                Xsub[c] = np.log1p(np.clip(Xsub[c].astype(float).fillna(0.0), a_min=0, a_max=None))
            else:
                Xsub[c] = Xsub[c].astype(float).fillna(0.0)
        return Xsub.values


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='traffic_dataset.csv')
    p.add_argument('--out', default='baseline_results/cv')
    p.add_argument('--n-splits', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--jobs', type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if not os.path.exists(args.csv):
        print('CSV not found:', args.csv); return

    df = pd.read_csv(args.csv)
    feats = [c for c in DEFAULT_FEATURES if c in df.columns]
    print('Using features:', feats)

    df = df.dropna(subset=feats + ['label','source_file'])
    X = df[feats]
    y = df['label'].astype(int)
    groups = df['source_file'].values

    pre = DFPreprocessor(feature_names=feats, heavy_tail=HEAVY_TAIL_COLS)
    scaler = StandardScaler()

    # pipeline: preprocessor -> scaler -> classifier
    pipe = Pipeline([
        ('pre', pre),
        ('scaler', scaler),
        ('clf', RandomForestClassifier(random_state=args.seed, n_jobs=-1))
    ])

    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [8, 12],
        'clf__min_samples_leaf': [1, 5],
        # 'clf__class_weight': [None, 'balanced']
    }

    cv = GroupKFold(n_splits=args.n_splits)
    gs = GridSearchCV(pipe, param_grid, scoring='f1_macro', cv=cv, n_jobs=args.jobs, verbose=2, return_train_score=True)

    print('Starting GridSearchCV...')
    gs.fit(X, y, groups=groups)

    print('Grid search done. Best params:')
    print(pformat(gs.best_params_))

    # evaluate best estimator per-fold to get fold metrics
    best = gs.best_estimator_
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        # fit best on fold's train
        best.fit(Xtr, ytr)
        preds = best.predict(Xte)
        acc = float(accuracy_score(yte, preds))
        f1 = float(f1_score(yte, preds, average='macro'))
        fold_metrics.append({'fold': int(fold), 'acc': acc, 'f1_macro': f1, 'n_test': int(len(yte))})

    # aggregate
    accs = [m['acc'] for m in fold_metrics]
    f1s = [m['f1_macro'] for m in fold_metrics]
    summary = {
        'best_params': gs.best_params_,
        'cv_mean_acc': float(np.mean(accs)),
        'cv_std_acc': float(np.std(accs)),
        'cv_mean_f1_macro': float(np.mean(f1s)),
        'cv_std_f1_macro': float(np.std(f1s)),
        'folds': fold_metrics
    }

    with open(os.path.join(args.out, 'grid_search_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # save full cv results (only selected columns to keep file small)
    cvr = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in gs.cv_results_.items()}
    with open(os.path.join(args.out, 'cv_results_raw.json'), 'w') as f:
        json.dump(cvr, f, indent=2)

    # save best model
    joblib.dump(best, os.path.join(args.out, 'rf_cv_best.joblib'))

    print('Saved best model and grid search summary to', args.out)


if __name__ == '__main__':
    main()
