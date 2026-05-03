import os
import subprocess
import argparse
import math
import pandas as pd
import numpy as np
from scipy.stats import skew, entropy

DATASET_ROOT = "dataset"

LABEL_MAP = {
    "youtube": 0,
    "wikipedia": 1,
    "agario": 2
}

# Change this to the IP of the capture machine (the client) in your pcap captures.
# If you don't know it, set to None and the script will try to guess (basic heuristic).
CLIENT_IP = "100.72.115.16"

# If set (seconds), split flows when inter-packet gap exceeds this value.
FLOW_TIMEOUT = None


# -----------------------------
# PCAP → packet table
# -----------------------------
def load_pcap(pcap_path):
    cmd = [
        "tshark",
        "-r", pcap_path,
        "-T", "fields",
        "-E", "separator=,",
        "-E", "header=n",
        "-e", "frame.time_epoch",
        "-e", "ip.src",
        "-e", "ip.dst",
        "-e", "tcp.srcport",
        "-e", "tcp.dstport",
        "-e", "udp.srcport",
        "-e", "udp.dstport",
        "-e", "ip.proto",
        "-e", "frame.len"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    rows = []
    for line in result.stdout.splitlines():
        parts = line.split(",")
        if len(parts) != 9:
            continue
        rows.append(parts)

    return pd.DataFrame(rows, columns=[
        "time", "ip_src", "ip_dst",
        "tcp_sport", "tcp_dport",
        "udp_sport", "udp_dport",
        "proto", "length"
    ])


# -----------------------------
# cleaning
# -----------------------------
def clean_df(df):
    df = df.dropna(subset=["ip_src", "ip_dst"])

    # FIX: convert empty strings to NaN first
    df = df.replace("", np.nan)

    # Convert numeric-ish columns safely
    df["tcp_sport"] = pd.to_numeric(df["tcp_sport"], errors="coerce")
    df["tcp_dport"] = pd.to_numeric(df["tcp_dport"], errors="coerce")
    df["udp_sport"] = pd.to_numeric(df["udp_sport"], errors="coerce")
    df["udp_dport"] = pd.to_numeric(df["udp_dport"], errors="coerce")

    df["src_port"] = df["tcp_sport"].fillna(df["udp_sport"]).fillna(0).astype(int)
    df["dst_port"] = df["tcp_dport"].fillna(df["udp_dport"]).fillna(0).astype(int)

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["length"] = pd.to_numeric(df["length"], errors="coerce").fillna(0).astype(int)
    df["proto"] = pd.to_numeric(df["proto"], errors="coerce").fillna(0).astype(int)

    return df


# -----------------------------
# flow grouping key
# -----------------------------
def add_flow_id(df):
    def flow_key(r):
        ip1, ip2 = sorted([r["ip_src"], r["ip_dst"]])

        # preserve direction info instead of sorting ports
        if r["ip_src"] <= r["ip_dst"]:
            port1, port2 = int(r["src_port"]), int(r["dst_port"])
        else:
            port1, port2 = int(r["dst_port"]), int(r["src_port"])

        proto = int(r["proto"])

        return (ip1, ip2, port1, port2, proto)

    df["flow_key"] = df.apply(flow_key, axis=1)
    return df
    
# -----------------------------
# flow features
# -----------------------------
def extract_flow_features(df):
    features = []

    # Use the global CLIENT_IP by default; caller can override by setting df._client_ip
    client_ip = getattr(df, "_client_ip", None) or CLIENT_IP

    # burst tau in milliseconds -> seconds
    burst_tau_ms = getattr(df, "_burst_tau_ms", 50)
    burst_tau = float(burst_tau_ms) / 1000.0

    flow_timeout = getattr(df, "_flow_timeout", None)
    if flow_timeout is not None and flow_timeout <= 0:
        flow_timeout = None

    # If client_ip is None, try a simple heuristic: the most frequent IP overall
    if client_ip is None:
        most_common = pd.concat([df["ip_src"], df["ip_dst"]]).value_counts()
        if len(most_common) > 0:
            client_ip = most_common.index[0]

    def compute_features(g, flow_id):
        g = g.sort_values("time").reset_index(drop=True)

        sizes = g["length"].values.astype(int)
        times = g["time"].values.astype(float)

        # flow duration
        flow_duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0

        # inter-arrival times
        if len(times) > 1:
            iats = np.diff(times)
            iat_mean = float(np.mean(iats))
            iat_std = float(np.std(iats))
            iat_min = float(np.min(iats))
            iat_max = float(np.max(iats))
            iat_median = float(np.median(iats))
        else:
            iats = np.array([])
            iat_mean = iat_std = iat_min = iat_max = iat_median = 0.0

        # per-direction IATs and stats helper (computed after direction masks below)
        def dir_iat_stats(mask):
            # mask selects packets in given direction; compute iats between successive packets of that direction
            idxs = np.where(mask)[0]
            if len(idxs) <= 1:
                return {
                    "q10": np.nan, "q50": np.nan, "q90": np.nan,
                    "cv": np.nan, "skew": np.nan
                }
            t = times[idxs]
            diats = np.diff(t)
            q10, q50, q90 = np.percentile(diats, [10, 50, 90])
            mean = np.mean(diats)
            std = np.std(diats)
            cv = float(std / (mean + 1e-12))
            sk = float(skew(diats)) if len(diats) > 2 else 0.0
            return {"q10": float(q10), "q50": float(q50), "q90": float(q90), "cv": float(cv), "skew": float(sk)}

        # direction per packet: 1 for upload (client->server), -1 for download
        if client_ip is not None:
            dirs = np.where(g["ip_src"].values == client_ip, 1, -1)
        else:
            # fallback: assume first packet src is client
            first_src = g.iloc[0]["ip_src"]
            dirs = np.where(g["ip_src"].values == first_src, 1, -1)

        # upload/download split
        upload_mask = dirs == 1
        download_mask = dirs == -1

        # now compute per-direction IAT stats
        up_iat = dir_iat_stats(upload_mask)
        down_iat = dir_iat_stats(download_mask)

        upload_bytes = int(sizes[upload_mask].sum())
        download_bytes = int(sizes[download_mask].sum())

        upload_packets = int(np.sum(upload_mask))
        download_packets = int(np.sum(download_mask))

        total_bytes = upload_bytes + download_bytes
        total_packets = upload_packets + download_packets

        total_bytes_eps = total_bytes if total_bytes > 0 else 1e-6
        total_packets_eps = total_packets if total_packets > 0 else 1e-6

        # packet-size percentiles
        if len(sizes) > 0:
            q10, q25, q50, q75, q90 = np.percentile(sizes, [10, 25, 50, 75, 90])
            pkt_size_mean = float(np.mean(sizes))
            pkt_size_std = float(np.std(sizes))
            frac_small_pkts = float((sizes < 100).sum()) / float(len(sizes))
        else:
            q10 = q25 = q50 = q75 = q90 = 0.0
            pkt_size_mean = pkt_size_std = frac_small_pkts = 0.0

        # first-N packet size stats per-direction (N=4)
        N_first = 4
        def first_n_stats(mask):
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                return {"mean": np.nan, "std": np.nan}
            first_idxs = idxs[:N_first]
            vals = sizes[first_idxs]
            return {"mean": float(vals.mean()), "std": float(vals.std())}

        first_up = first_n_stats(upload_mask)
        first_down = first_n_stats(download_mask)

        # packet-size histogram buckets and entropy (global and per-direction)
        bins = [0, 200, 600, 1200, np.inf]
        try:
            hist_all, _ = np.histogram(sizes, bins=bins)
            p_all = hist_all.astype(float) / (hist_all.sum() + 1e-12)
            sz_entropy = float(entropy(p_all))
        except Exception:
            p_all = np.zeros(len(bins)-1)
            sz_entropy = 0.0

        def hist_props(mask):
            try:
                arr = sizes[mask]
                h, _ = np.histogram(arr, bins=bins)
                p = h.astype(float) / (h.sum() + 1e-12)
                ent = float(entropy(p))
                return p.tolist(), ent
            except Exception:
                return [0.0]*(len(bins)-1), 0.0

        p_up, ent_up = hist_props(upload_mask)
        p_down, ent_down = hist_props(download_mask)

        # burst detection (base tau and multi-scale variants)
        # base burst using burst_tau (from CLI / df._burst_tau_ms)
        if len(times) > 0:
            dt = np.concatenate(([0.0], np.diff(times)))
            prev_dir = np.concatenate(([dirs[0]], dirs[:-1]))
            new_burst = (dt > burst_tau) | (dirs != prev_dir)
            burst_id = np.cumsum(new_burst)

            burst_df = pd.DataFrame({
                "burst_id": burst_id,
                "bytes": sizes,
                "packets": 1,
                "dir": dirs
            })

            burst_groups = burst_df.groupby("burst_id")
            burst_count = int(burst_groups.size().shape[0])
            burst_packets = burst_groups["packets"].sum().values
            burst_bytes = burst_groups["bytes"].sum().values

            avg_burst_packets = float(burst_packets.mean()) if len(burst_packets) > 0 else 0.0
            avg_burst_bytes = float(burst_bytes.mean()) if len(burst_bytes) > 0 else 0.0
            max_burst_packets = int(burst_packets.max()) if len(burst_packets) > 0 else 0
            max_burst_bytes = int(burst_bytes.max()) if len(burst_bytes) > 0 else 0
            frac_bytes_in_bursts = float(burst_bytes.sum()) / float(total_bytes_eps)

            # burst dispersion stats
            burst_bytes_cv = float(np.std(burst_bytes) / (np.mean(burst_bytes) + 1e-12)) if len(burst_bytes) > 0 else np.nan
            burst_packets_cv = float(np.std(burst_packets) / (np.mean(burst_packets) + 1e-12)) if len(burst_packets) > 0 else np.nan

            # burst inter-arrival intervals (between burst starts)
            if burst_count > 0:
                try:
                    burst_starts = burst_groups.apply(lambda d: times[d.index.min()])
                    burst_starts_vals = np.array(list(burst_starts))
                    if len(burst_starts_vals) > 1:
                        burst_iais = np.diff(burst_starts_vals)
                        burst_iai_cv = float(np.std(burst_iais) / (np.mean(burst_iais) + 1e-12))
                    else:
                        burst_iai_cv = np.nan
                except Exception:
                    burst_iai_cv = np.nan
            else:
                burst_iai_cv = np.nan

            # per-direction burst counts already have a download variant later; compute ratio
            burst_count_download = int(burst_df[burst_df['dir'] == -1].groupby('burst_id').size().shape[0]) if len(times) > 0 else 0
            burst_count_ratio_download = float(burst_count_download) / float(burst_count) if burst_count > 0 else 0.0

            # multi-scale burst features: try tau = 20ms, 50ms (base), 200ms
            multi_taus_ms = [20, int(burst_tau_ms), 200]
            multi_stats = {}
            for t_ms in multi_taus_ms:
                t = float(t_ms) / 1000.0
                dtt = np.concatenate(([0.0], np.diff(times)))
                newb = (dtt > t) | (dirs != np.concatenate(([dirs[0]], dirs[:-1])))
                bid = np.cumsum(newb)
                bdf = pd.DataFrame({'burst_id': bid, 'bytes': sizes, 'packets': 1, 'dir': dirs})
                bg = bdf.groupby('burst_id')
                bcount = int(bg.size().shape[0])
                bbytes = bg['bytes'].sum().values
                bavg = float(bbytes.mean()) if len(bbytes) > 0 else 0.0
                bmax = int(bbytes.max()) if len(bbytes) > 0 else 0
                bfrac = float(bbytes.sum()) / float(total_bytes_eps)
                multi_stats[t_ms] = (bcount, bavg, bmax, bfrac)
        else:
            burst_count = avg_burst_packets = avg_burst_bytes = 0.0
            max_burst_packets = max_burst_bytes = 0
            frac_bytes_in_bursts = 0.0
            burst_iai_cv = np.nan
            burst_bytes_cv = np.nan
            burst_packets_cv = np.nan
            burst_count_download = 0
            burst_count_ratio_download = 0.0
            multi_stats = {20: (0,0.0,0,0.0), int(burst_tau_ms): (0,0.0,0,0.0), 200: (0,0.0,0,0.0)}

        # peak rate approximation using 1s bins (floor of timestamp)
        if len(times) > 0:
            bins = np.floor(times).astype(int)
            bin_bytes = {}
            bin_bytes_download = {}
            bin_bytes_upload = {}
            for b, s, d in zip(bins, sizes, dirs):
                bin_bytes[b] = bin_bytes.get(b, 0) + int(s)
                if d == -1:
                    bin_bytes_download[b] = bin_bytes_download.get(b, 0) + int(s)
                else:
                    bin_bytes_upload[b] = bin_bytes_upload.get(b, 0) + int(s)
            peak_rate_1s = float(max(bin_bytes.values())) if len(bin_bytes) > 0 else 0.0
            peak_download_1s = float(max(bin_bytes_download.values())) if len(bin_bytes_download) > 0 else 0.0
            peak_upload_1s = float(max(bin_bytes_upload.values())) if len(bin_bytes_upload) > 0 else 0.0
        else:
            peak_rate_1s = 0.0
            peak_download_1s = 0.0
            peak_upload_1s = 0.0

        return {
            "flow_id": str(flow_id),

            # existing features
            "packet_count": int(len(g)),
            "byte_count": int(np.sum(sizes)),

            # NEW signals
            "upload_bytes": upload_bytes,
            "download_bytes": download_bytes,

            "upload_packets": upload_packets,
            "download_packets": download_packets,

            "byte_ratio_up_down": float(upload_bytes) / float(total_bytes_eps),
            "packet_ratio_up_down": float(upload_packets) / float(total_packets_eps),

            "direction_imbalance": abs(upload_bytes - download_bytes) / float(total_bytes_eps),

            # timing
            "flow_duration": float(flow_duration),
            "iat_mean": float(iat_mean),
            "iat_std": float(iat_std),
            "iat_min": float(iat_min),
            "iat_max": float(iat_max),
            "iat_median": float(iat_median),

            # packet sizes
            "pkt_size_mean": pkt_size_mean,
            "pkt_size_std": pkt_size_std,
            "pkt_q10": float(q10),
            "pkt_q25": float(q25),
            "pkt_q50": float(q50),
            "pkt_q75": float(q75),
            "pkt_q90": float(q90),
            "frac_small_pkts": frac_small_pkts,

            # first-N packet stats per-direction
            "first4_up_mean": first_up["mean"],
            "first4_up_std": first_up["std"],
            "first4_down_mean": first_down["mean"],
            "first4_down_std": first_down["std"],

            # packet-size histogram proportions and entropy
            "pkt_bin_p0": float(p_all[0]) if len(p_all)>0 else 0.0,
            "pkt_bin_p1": float(p_all[1]) if len(p_all)>1 else 0.0,
            "pkt_bin_p2": float(p_all[2]) if len(p_all)>2 else 0.0,
            "pkt_bin_p3": float(p_all[3]) if len(p_all)>3 else 0.0,
            "pkt_size_entropy": float(sz_entropy),

            "pkt_bin_up_p0": float(p_up[0]) if len(p_up)>0 else 0.0,
            "pkt_bin_up_p1": float(p_up[1]) if len(p_up)>1 else 0.0,
            "pkt_bin_up_p2": float(p_up[2]) if len(p_up)>2 else 0.0,
            "pkt_bin_up_p3": float(p_up[3]) if len(p_up)>3 else 0.0,
            "pkt_bin_down_p0": float(p_down[0]) if len(p_down)>0 else 0.0,
            "pkt_bin_down_p1": float(p_down[1]) if len(p_down)>1 else 0.0,
            "pkt_bin_down_p2": float(p_down[2]) if len(p_down)>2 else 0.0,
            "pkt_bin_down_p3": float(p_down[3]) if len(p_down)>3 else 0.0,

            "pkt_size_entropy_up": float(ent_up),
            "pkt_size_entropy_down": float(ent_down),

            # per-direction IAT quantiles/CV/skew
            "iat_up_q10": up_iat["q10"],
            "iat_up_median": up_iat["q50"],
            "iat_up_q90": up_iat["q90"],
            "iat_up_cv": up_iat["cv"],
            "iat_up_skew": up_iat["skew"],

            "iat_down_q10": down_iat["q10"],
            "iat_down_median": down_iat["q50"],
            "iat_down_q90": down_iat["q90"],
            "iat_down_cv": down_iat["cv"],
            "iat_down_skew": down_iat["skew"],

            # burst periodicity
            "burst_iai_cv": float(burst_iai_cv) if burst_iai_cv is not None else np.nan,

            # bursts
            "burst_count": int(burst_count),
            "avg_burst_packets": float(avg_burst_packets),
            "avg_burst_bytes": float(avg_burst_bytes),
            "max_burst_packets": int(max_burst_packets),
            "max_burst_bytes": int(max_burst_bytes),
            "frac_bytes_in_bursts": float(frac_bytes_in_bursts),

            # extra burst dispersion / ratio features
            "burst_bytes_cv": float(burst_bytes_cv) if not np.isnan(burst_bytes_cv) else np.nan,
            "burst_packets_cv": float(burst_packets_cv) if not np.isnan(burst_packets_cv) else np.nan,
            "burst_count_ratio_download": float(burst_count_ratio_download),

            # multi-tau burst features (suffix _tXX where XX is ms)
            "burst_count_t20": int(multi_stats.get(20,(0,0,0,0))[0]),
            "avg_burst_bytes_t20": float(multi_stats.get(20,(0,0,0,0))[1]),
            "max_burst_bytes_t20": int(multi_stats.get(20,(0,0,0,0))[2]),
            "frac_bytes_in_bursts_t20": float(multi_stats.get(20,(0,0,0,0))[3]),

            f"burst_count_t{int(burst_tau_ms)}": int(multi_stats.get(int(burst_tau_ms),(0,0,0,0))[0]),
            f"avg_burst_bytes_t{int(burst_tau_ms)}": float(multi_stats.get(int(burst_tau_ms),(0,0,0,0))[1]),
            f"max_burst_bytes_t{int(burst_tau_ms)}": int(multi_stats.get(int(burst_tau_ms),(0,0,0,0))[2]),
            f"frac_bytes_in_bursts_t{int(burst_tau_ms)}": float(multi_stats.get(int(burst_tau_ms),(0,0,0,0))[3]),

            "burst_count_t200": int(multi_stats.get(200,(0,0,0,0))[0]),
            "avg_burst_bytes_t200": float(multi_stats.get(200,(0,0,0,0))[1]),
            "max_burst_bytes_t200": int(multi_stats.get(200,(0,0,0,0))[2]),
            "frac_bytes_in_bursts_t200": float(multi_stats.get(200,(0,0,0,0))[3]),

            # rates
            "peak_rate_1s": float(peak_rate_1s),
            "peak_download_1s": float(peak_download_1s),
            "peak_upload_1s": float(peak_upload_1s),
            # direction-specific burst stats (download)
            "burst_count_download": int(burst_df[burst_df['dir'] == -1].groupby('burst_id').size().shape[0]) if len(times) > 0 else 0,
            "avg_burst_bytes_download": float(burst_df[burst_df['dir'] == -1].groupby('burst_id')["bytes"].sum().mean()) if len(times) > 0 and len(burst_df[burst_df['dir'] == -1])>0 else 0.0,
            "max_burst_bytes_download": int(burst_df[burst_df['dir'] == -1].groupby('burst_id')["bytes"].sum().max()) if len(times) > 0 and len(burst_df[burst_df['dir'] == -1])>0 else 0,
        }

    for flow_id, g in df.groupby("flow_key"):
        if flow_timeout is None:
            features.append(compute_features(g, flow_id))
            continue

        g_sorted = g.sort_values("time").reset_index(drop=True)
        times = g_sorted["time"].values.astype(float)

        if len(times) <= 1:
            features.append(compute_features(g_sorted, f"{flow_id}_seg0"))
            continue

        split_points = np.where(np.diff(times) > flow_timeout)[0]
        start = 0
        seg_idx = 0
        for sp in split_points:
            end = sp + 1
            seg = g_sorted.iloc[start:end]
            features.append(compute_features(seg, f"{flow_id}_seg{seg_idx}"))
            seg_idx += 1
            start = end

        if start < len(g_sorted):
            seg = g_sorted.iloc[start:]
            features.append(compute_features(seg, f"{flow_id}_seg{seg_idx}"))

    return pd.DataFrame(features)


# -----------------------------
# process single PCAP
# -----------------------------
def process_pcap(path, label):
    df = load_pcap(path)
    df = clean_df(df)
    df._client_ip = CLIENT_IP
    df._burst_tau_ms = getattr(df, "_burst_tau_ms", 50)
    df._flow_timeout = FLOW_TIMEOUT
    df = add_flow_id(df)

    flow_df = extract_flow_features(df)
    flow_df["label"] = label

    return flow_df


# -----------------------------
# MAIN DATASET BUILDER (UPDATED)
# -----------------------------
def build_dataset(n_per_class=3):
    all_data = []

    for label_name, label_id in LABEL_MAP.items():
        folder = os.path.join(DATASET_ROOT, label_name)

        # sort so "first N" is deterministic (based on filename timestamps)
        files = sorted([
            f for f in os.listdir(folder)
            if f.endswith(".pcapng")
        ])

        selected_files = files[:n_per_class]

        print(f"\nClass: {label_name}")
        print(f"Using {len(selected_files)} files")

        for file in selected_files:
            path = os.path.join(folder, file)

            try:
                df = process_pcap(path, label_id)
                df["source_file"] = file
                all_data.append(df)

                print(f"  processed {file} -> {len(df)} flows")

            except Exception as e:
                print(f"  failed {file}: {e}")

    final_df = pd.concat(all_data, ignore_index=True)

    return final_df


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build traffic dataset from PCAPNG files")
    parser.add_argument("--n-per-class", type=int, default=3, help="number of PCAP files per class to process")
    parser.add_argument("--client-ip", type=str, default=None, help="client IP used to determine upload/download direction")
    parser.add_argument("--burst-ms", type=int, default=50, help="burst detection tau in milliseconds")
    parser.add_argument("--flow-timeout", type=float, default=None, help="split flows if inter-packet gap exceeds this many seconds")
    parser.add_argument("--out", type=str, default="traffic_dataset.csv", help="output CSV path")

    args = parser.parse_args()

    # set global CLIENT_IP if provided
    if args.client_ip:
        CLIENT_IP = args.client_ip
    if args.flow_timeout is not None:
        FLOW_TIMEOUT = args.flow_timeout

    dataset = build_dataset(n_per_class=args.n_per_class)

    # attach df-level attributes used by extract_flow_features
    # (we attach to the concatenated result for convenience)
    dataset._client_ip = CLIENT_IP
    dataset._burst_tau_ms = args.burst_ms
    dataset._flow_timeout = FLOW_TIMEOUT

    print("\nFinal dataset shape:", dataset.shape)
    print(dataset.groupby("label").size())

    out_path = args.out
    if args.out == "traffic_dataset.csv" and args.flow_timeout is not None:
        timeout_label = str(args.flow_timeout).rstrip("0").rstrip(".")
        out_path = f"traffic_dataset_timeout{timeout_label}s.csv"

    dataset.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")