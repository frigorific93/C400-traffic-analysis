#!/usr/bin/env python3
"""Predict labels for flows extracted from a .pcapng file.

Usage: python3 predict_pcap.py --pcap input.pcapng --model baseline_results/lgbm_newfeatures/model.joblib --out predictions.csv

The script mirrors the dataset building pipeline in `build_dataset.py` and the
preprocessing used for LightGBM evaluation (DFPreprocessor from cv_baseline.py).
If a joblib model isn't available for LightGBM, you can point `--model` to any
scikit-learn compatible estimator saved with joblib.
"""

import argparse
import os
import joblib
import pandas as pd
import numpy as np
import importlib.util
import sys

# Prefer loading local submission scripts by path so the submission folder is
# runnable when extracted (don't rely on 'submission' being an importable package).
SUB_ROOT = os.path.abspath(os.path.dirname(__file__))

def _load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Prefer to use the project's top-level modules if available (they contain
# the full feature extraction used during training). If those imports fail,
# fall back to the lightweight local copies in submission/scripts/.
try:
    import importlib
    _build_mod = importlib.import_module('build_dataset')
    _cv_mod = importlib.import_module('cv_baseline')
    # mark that we used project modules
    used_root_modules = True
except Exception:
    # fall back to submission/scripts
    _build_path = os.path.join(SUB_ROOT, 'scripts', 'build_dataset.py')
    _cv_path = os.path.join(SUB_ROOT, 'scripts', 'cv_baseline.py')
    if not os.path.exists(_build_path) or not os.path.exists(_cv_path):
        raise FileNotFoundError('Required helper scripts missing: submission/scripts/build_dataset.py or cv_baseline.py')
    _build_mod = _load_module_from_file('submission_build_dataset', _build_path)
    _cv_mod = _load_module_from_file('submission_cv_baseline', _cv_path)
    used_root_modules = False

# Make the loaded modules importable under the original names so pickled
# pipelines referencing 'cv_baseline' or 'build_dataset' can be unpickled.
sys.modules.setdefault('cv_baseline', _cv_mod)
sys.modules.setdefault('build_dataset', _build_mod)

process_pcap = _build_mod.process_pcap
DFPreprocessor = _cv_mod.DFPreprocessor
HEAVY_TAIL_COLS = getattr(_cv_mod, 'HEAVY_TAIL_COLS', [])
DEFAULT_FEATURES = getattr(_cv_mod, 'DEFAULT_FEATURES', [])


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)


def main(pcap, model_path, out_csv, client_ip=None, burst_ms=50, flow_timeout=60.0, debug=False):
    # Extract flows from pcap
    df = process_pcap(pcap, label=None)

    # process_pcap attaches features and returns flow_df; drop "label" if present
    if 'label' in df.columns:
        df = df.drop(columns=['label'])

    # optionally override client ip / burst tau / flow timeout if provided
    if client_ip:
        df._client_ip = client_ip
    df._burst_tau_ms = burst_ms
    df._flow_timeout = flow_timeout

    # load model
    model = load_model(model_path)

    if debug:
        print('Loaded model:', type(model), getattr(model, '__class__', None))
        try:
            print('Model repr:', repr(model)[:1000])
        except Exception:
            pass

    # Build feature list: prefer DEFAULT_FEATURES but include extras present in df
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
    if len(feats) == 0:
        # save the extracted dataframe for debugging
        debug_path = os.path.join(SUB_ROOT, 'last_failed_extract.csv')
        try:
            df.to_csv(debug_path, index=False)
            print(f'No matching features found. Saved extracted flow table to {debug_path}')
            print('Extracted columns:', list(df.columns))
            print('First 5 rows:')
            print(df.head().to_string())
        except Exception:
            print('No features found and failed to save debug CSV')
        raise RuntimeError('No features found in pcap extract that match expected features')

    X = df[feats]

    if debug:
        print('\n--- Debug: feature checklist ---')
        print('Number of flows extracted:', len(df))
        print('Features used (count):', len(feats))
        print('Feature names:', feats)
        # show dtypes and missing counts
        dtypes = X.dtypes.astype(str).to_dict()
        nulls = X.isnull().sum().to_dict()
        print('Feature dtypes (sample):', {k: dtypes.get(k) for k in list(feats)[:10]})
        print('Null counts (sample):', {k: nulls.get(k) for k in list(feats)[:10]})
        try:
            print('Feature sample stats:')
            print(X.describe().transpose().loc[feats[:10], ['count','mean','std','min','max']])
        except Exception:
            pass

    # If the loaded model is a sklearn Pipeline (has named_steps or steps),
    # assume it includes preprocessing and scaling and pass the raw DataFrame.
    is_pipeline = hasattr(model, 'named_steps') or hasattr(model, 'steps')

    X_to_pred = None
    try:
        if is_pipeline:
            # If pipeline, attempt to inspect expected features
            if debug:
                try:
                    if hasattr(model, 'named_steps'):
                        print('Pipeline steps:', list(model.named_steps.keys()))
                except Exception:
                    pass

            try:
                preds = model.predict(X)
                X_to_pred = X
            except KeyError as ke:
                # sklearn pipeline may expect columns not present in X. Try to
                # recover by adding missing columns with zeros and retry once.
                msg = str(ke)
                if debug:
                    print('Pipeline KeyError message:', msg)
                # attempt to parse missing columns from message like "[...] not in index"
                if 'not in index' in msg:
                    import re
                    m = re.search(r"\[([^\]]+)\] not in index", msg)
                    if m:
                        missing = [c.strip().strip("'\"") for c in m.group(1).split(',')]
                    else:
                        missing = []
                else:
                    missing = []

                if debug:
                    print('Missing columns detected, padding with zeros:', missing)

                for c in missing:
                    if c not in X.columns:
                        X[c] = 0

                preds = model.predict(X)
                X_to_pred = X
        else:
            pre = DFPreprocessor(feature_names=feats, heavy_tail=HEAVY_TAIL_COLS)
            X_arr = pre.transform(X)
            if debug:
                print('Transformed feature array shape:', getattr(X_arr, 'shape', None))

            # Wrap transformed array into DataFrame with feature names so models
            # that expect feature names (LightGBM) get the correct columns.
            try:
                X_arr_df = pd.DataFrame(X_arr, columns=feats)
            except Exception:
                # fallback: create DataFrame without column names
                X_arr_df = pd.DataFrame(X_arr)

            # Try to detect expected feature names from LightGBM / sklearn wrappers
            expected = None
            try:
                if hasattr(model, 'booster_') and model.booster_ is not None:
                    expected = list(model.booster_.feature_name())
                elif hasattr(model, 'feature_name_'):
                    expected = list(getattr(model, 'feature_name_'))
            except Exception:
                expected = None

            if expected:
                # Reindex to expected order; pad missing cols with zeros
                missing = [c for c in expected if c not in X_arr_df.columns]
                extra = [c for c in X_arr_df.columns if c not in expected]
                if len(missing) > 0:
                    if debug:
                        print('Padding missing features with zeros:', missing)
                    for c in missing:
                        X_arr_df[c] = 0.0
                # reindex to expected ordering
                X_arr_df = X_arr_df.reindex(columns=expected, fill_value=0.0)
                if debug:
                    print('Reindexed DataFrame to model expected features. Extra dropped:', extra)
                X_to_pred = X_arr_df
            else:
                # If no expected names found, pass the DataFrame (has column names = feats)
                X_to_pred = X_arr_df

            if debug:
                print('\n--- Debug BEFORE predict (non-pipeline) ---')
                print('X_to_pred type:', type(X_to_pred))
                try:
                    print('X_to_pred columns:', list(X_to_pred.columns))
                except Exception:
                    try:
                        print('X_to_pred shape:', getattr(X_to_pred, 'shape', None))
                    except Exception:
                        pass
                try:
                    if hasattr(model, 'booster_') and model.booster_ is not None:
                        print('Model booster feature names (first 20):', list(model.booster_.feature_name())[:20])
                except Exception:
                    pass

            preds = model.predict(X_to_pred)
            X_to_pred = X_to_pred
    except Exception as e:
        # add context in debug mode
        if debug:
            import traceback
            traceback.print_exc()
        raise RuntimeError(f'Failed to run model.predict: {e}')

    # If possible, also compute probabilities and include them in output for debugging
    probs = None
    try:
        if hasattr(model, 'predict_proba'):
            if X_to_pred is not None:
                probs = model.predict_proba(X_to_pred)
            else:
                # fallback
                probs = model.predict_proba(X if is_pipeline else X_arr)
    except Exception:
        probs = None

    # map numeric labels back to names if possible (try to read label map from baseline scripts)
    LABEL_MAP = {0: 'youtube', 1: 'wikipedia', 2: 'agario'}
    pred_names = [LABEL_MAP.get(int(p), str(p)) for p in preds]

    out_df = df.copy()
    out_df['pred_label'] = preds
    out_df['pred_name'] = pred_names

    out_df.to_csv(out_csv, index=False)
    print('Wrote predictions to', out_csv)
    # Produce an overall source prediction by majority vote of per-flow predictions
    try:
        overall_counts = out_df['pred_name'].value_counts()
        if len(overall_counts) > 0:
            top_name = overall_counts.idxmax()
            top_count = int(overall_counts.max())
            total = int(overall_counts.sum())
            print(f'Overall predicted source: {top_name} ({top_count}/{total} flows)')
            print('Counts:')
            for name, cnt in overall_counts.items():
                print(f'  {name}: {int(cnt)}')
    except Exception:
        pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('pcap', help='input filename inside submission/pcaps (e.g. example_youtube.pcapng)')
    p.add_argument('--model', default='lgbm_inference.joblib', help='relative model path inside submission (defaults to lgbm_inference.joblib)')
    p.add_argument('--out', default='pcap_predictions.csv', help='output CSV')
    p.add_argument('--client-ip', default=None, help='client IP to determine upload/download direction')
    p.add_argument('--burst-ms', type=int, default=50, help='burst detection tau (ms)')
    p.add_argument('--flow-timeout', type=float, default=60.0, help='flow inactivity timeout (s)')
    p.add_argument('--debug', action='store_true', help='enable debug output')
    args = p.parse_args()

    # Build absolute paths relative to submission/ root so script runs inside extracted submission/
    SUB_ROOT = os.path.abspath(os.path.dirname(__file__))
    # If the user provided a path (contains os.sep) or absolute path, accept it as-is
    if os.path.isabs(args.pcap):
        pcap_path = args.pcap
    elif os.path.sep in args.pcap:
        # treat relative paths as relative to the submission root
        pcap_path = os.path.join(SUB_ROOT, args.pcap)
    else:
        pcap_path = os.path.join(SUB_ROOT, 'pcaps', args.pcap)
    model_path = os.path.join(SUB_ROOT, args.model)

    # If model isn't found at the provided path, also check submission/models/
    if not os.path.exists(model_path):
        alt_path = os.path.join(SUB_ROOT, 'models', os.path.basename(args.model))
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(
                f'Model not found: tried {os.path.join(SUB_ROOT, args.model)} and {alt_path}.\n'
                f'Place your trained pipeline at either submission/{os.path.basename(args.model)} or submission/models/{os.path.basename(args.model)}'
            )

    out_csv = os.path.join(SUB_ROOT, args.out)

    main(pcap_path, model_path, out_csv, client_ip=args.client_ip, burst_ms=args.burst_ms, flow_timeout=args.flow_timeout, debug=args.debug)
