#!/usr/bin/env python3
"""
Extract instantaneous magnetic-field magnitude time series from LFA Kassiopeia
ROOT output files, looping over one sub-directory per event.

Each event sub-directory of --results-dir is expected to contain a ROOT file
matching KassCavity*Output.root with a "component_step_world_DATA" tree
holding "time", "magnetic_field_x", "magnetic_field_y", "magnetic_field_z".
The "time" branch (and therefore --t-min/--t-max and all t_start_s/t_end_s
output) is in seconds.

Two modes:
  average    - time-weighted average |B| over an (optionally windowed) time
               range (in seconds) for every event, written to a single
               summary CSV.
  timeseries - dump the full (time, |B|) arrays for every event, one .npz
               file per event written to --output-dir.

Examples:
  python extract_lfa_b_field.py --results-dir /path/to/results \
      --mode average --output b_field_averages.csv
  python extract_lfa_b_field.py --results-dir /path/to/results \
      --mode average --t-min 0 --t-max 1e-4 --output b_field_averages_100us.csv
  python extract_lfa_b_field.py --results-dir /path/to/results \
      --mode timeseries --output-dir ./b_timeseries
  python extract_lfa_b_field.py --results-dir /path/to/results \
      --mode average --limit 1 --output /tmp/test.csv
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import uproot

TREE_NAME = "component_step_world_DATA"
ROOT_GLOB_PATTERN = "KassCavity*Output.root"
BRANCHES = ["time", "magnetic_field_x", "magnetic_field_y", "magnetic_field_z"]

# Event trees here can hold tens of millions of rows, and this is often run
# on login nodes with a hard per-session memory cap (e.g. cgroup-limited to
# 1 GiB). Everything below reads the ROOT file in bounded-size chunks
# instead of loading whole branches at once, so peak memory stays small
# regardless of file size.
DEFAULT_STEP_SIZE = "20 MB"


def find_root_file(event_dir):
    matches = sorted(glob.glob(os.path.join(event_dir, ROOT_GLOB_PATTERN)))
    if not matches:
        return None
    if len(matches) > 1:
        print(
            f"  warning: {len(matches)} files match {ROOT_GLOB_PATTERN} in "
            f"{event_dir}, using {os.path.basename(matches[0])}",
            file=sys.stderr,
        )
    return matches[0]


def get_num_entries(root_path):
    with uproot.open(root_path) as f:
        return f[TREE_NAME].num_entries


def iterate_time_and_b(root_path, step_size=DEFAULT_STEP_SIZE):
    """Yield (time, |B|) chunk by chunk without holding the full tree in memory."""
    with uproot.open(root_path) as f:
        tree = f[TREE_NAME]
        for chunk in tree.iterate(BRANCHES, step_size=step_size, library="np"):
            t = chunk["time"]
            b = np.sqrt(
                chunk["magnetic_field_x"] ** 2
                + chunk["magnetic_field_y"] ** 2
                + chunk["magnetic_field_z"] ** 2
            )
            yield t, b


def time_weighted_average(root_path, t_min=None, t_max=None, step_size=DEFAULT_STEP_SIZE):
    integral = 0.0
    n_total = 0
    t_start_global = None
    t_end_global = None
    last_t = last_b = None

    for t, b in iterate_time_and_b(root_path, step_size):
        if t_max is not None and t.size and t[0] > t_max:
            break  # time is monotonic increasing, nothing left in range

        if t_min is not None or t_max is not None:
            mask = np.ones(t.shape, dtype=bool)
            if t_min is not None:
                mask &= t >= t_min
            if t_max is not None:
                mask &= t <= t_max
            t, b = t[mask], b[mask]

        if t.size == 0:
            continue

        if last_t is not None:
            t_ext = np.concatenate(([last_t], t))
            b_ext = np.concatenate(([last_b], b))
        else:
            t_ext, b_ext = t, b

        if t_ext.size >= 2:
            integral += np.trapezoid(b_ext, t_ext)

        last_t, last_b = t[-1], b[-1]
        n_total += t.size
        if t_start_global is None:
            t_start_global = t[0]
        t_end_global = t[-1]

    if n_total < 2 or t_start_global is None or t_end_global == t_start_global:
        return np.nan, n_total, t_start_global, t_end_global

    b_avg = integral / (t_end_global - t_start_global)
    return b_avg, n_total, t_start_global, t_end_global


def dump_timeseries(root_path, out_path, step_size=DEFAULT_STEP_SIZE):
    n_entries = get_num_entries(root_path)
    time_full = np.empty(n_entries, dtype=np.float64)
    b_full = np.empty(n_entries, dtype=np.float64)

    cursor = 0
    for t, b in iterate_time_and_b(root_path, step_size):
        n = t.size
        time_full[cursor:cursor + n] = t
        b_full[cursor:cursor + n] = b
        cursor += n

    time_full = time_full[:cursor]
    b_full = b_full[:cursor]
    np.savez_compressed(out_path, time=time_full, b_field=b_full)
    return cursor


def iter_event_dirs(results_dir, limit=None):
    entries = sorted(
        e for e in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, e))
    )
    if limit is not None:
        entries = entries[:limit]
    for entry in entries:
        yield entry, os.path.join(results_dir, entry)


def run_average(results_dir, output_path, t_min, t_max, limit, step_size=DEFAULT_STEP_SIZE):
    with open(output_path, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(
            ["event_id", "root_file", "b_avg_T", "n_points", "t_start_s", "t_end_s"]
        )

        for event_id, event_dir in iter_event_dirs(results_dir, limit):
            root_path = find_root_file(event_dir)
            if root_path is None:
                print(f"skip {event_id}: no {ROOT_GLOB_PATTERN} found", file=sys.stderr)
                continue

            print(f"processing {event_id} ({os.path.basename(root_path)})")
            try:
                b_avg, n_points, t_start, t_end = time_weighted_average(
                    root_path, t_min, t_max, step_size
                )
            except Exception as exc:
                print(f"  error: {exc}", file=sys.stderr)
                continue

            writer.writerow(
                [event_id, os.path.basename(root_path), b_avg, n_points, t_start, t_end]
            )
            out_f.flush()
            print(f"  B_avg = {b_avg:.10f} T  (n={n_points})")


def run_timeseries(results_dir, output_dir, limit, step_size=DEFAULT_STEP_SIZE):
    os.makedirs(output_dir, exist_ok=True)

    for event_id, event_dir in iter_event_dirs(results_dir, limit):
        root_path = find_root_file(event_dir)
        if root_path is None:
            print(f"skip {event_id}: no {ROOT_GLOB_PATTERN} found", file=sys.stderr)
            continue

        out_path = os.path.join(output_dir, f"{event_id}.npz")
        print(f"processing {event_id} ({os.path.basename(root_path)})")
        try:
            n_points = dump_timeseries(root_path, out_path, step_size)
        except Exception as exc:
            print(f"  error: {exc}", file=sys.stderr)
            continue

        print(f"  wrote {out_path} ({n_points} points)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        required=True,
        help="directory containing one sub-directory per event",
    )
    parser.add_argument(
        "--mode",
        choices=["average", "timeseries"],
        required=True,
        help="average: time-weighted average |B| per event; timeseries: dump full (time, |B|) arrays per event",
    )
    parser.add_argument(
        "--t-min", type=float, default=None,
        help="start of averaging window, in seconds (average mode only)",
    )
    parser.add_argument(
        "--t-max", type=float, default=None,
        help="end of averaging window, in seconds (average mode only)",
    )
    parser.add_argument(
        "--output", default="b_field_averages.csv",
        help="output CSV path (average mode only, default: b_field_averages.csv)",
    )
    parser.add_argument(
        "--output-dir", default="b_field_timeseries",
        help="output directory for per-event .npz files (timeseries mode only, default: b_field_timeseries)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only process the first N event directories (useful for testing)",
    )
    parser.add_argument(
        "--step-size", default=DEFAULT_STEP_SIZE,
        help=f"uproot chunk size for streaming reads, e.g. '20 MB' or '1000000' entries "
             f"(default: {DEFAULT_STEP_SIZE}); lower this if you hit memory limits",
    )
    args = parser.parse_args()

    if args.mode == "average":
        run_average(
            args.results_dir, args.output, args.t_min, args.t_max, args.limit, args.step_size
        )
    else:
        run_timeseries(args.results_dir, args.output_dir, args.limit, args.step_size)


if __name__ == "__main__":
    main()
