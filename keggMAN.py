#!/usr/bin/env python3
"""Summarize KEGG scan files by KO identifier across species."""

import argparse
import csv
import os
import sys
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser(description="Summarize KEGG scan results by KO across species files.")
    p.add_argument("folder", help="Folder containing KEGG scan files (tab-separated)")
    p.add_argument("-e", "--evalue", type=float, default=None,
                   help="E-value threshold (default: 1e-10 if -s not used)")
    p.add_argument("-s", "--significant", action="store_true",
                   help="Keep only entries marked with '*'")
    p.add_argument("-k", "--method", default="Kegg", help="ENTRY_TYPE value (default: Kegg)")
    p.add_argument("-o", "--output", default=None, help="Output file (default: stdout)")
    return p.parse_args()

def parse_file(filepath, evalue_thresh, use_sig):
    """Return dict {KO: set(protein_ids)} and {KO: description} for entries passing filters."""
    ko_proteins = defaultdict(set)
    ko_desc = {}

    with open(filepath, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            # skip comment lines
            if parts[0].startswith("#"):
                continue

            sig = parts[0].strip()
            # expect at least 7 columns: marker, gene, KO, thrshld, score, evalue, description
            if len(parts) < 7:
                continue

            gene = parts[1].strip()
            ko = parts[2].strip()
            evalue_str = parts[5].strip()
            desc = parts[6].strip().strip('"')

            if not ko or not gene:
                continue

            # apply significance filter
            if use_sig and sig != "*":
                continue

            # apply e-value filter
            if evalue_thresh is not None:
                if not evalue_str:
                    continue
                try:
                    ev = float(evalue_str)
                except ValueError:
                    continue
                if ev > evalue_thresh:
                    continue

            ko_proteins[ko].add(gene)
            if ko not in ko_desc:
                ko_desc[ko] = desc

    return ko_proteins, ko_desc

def main():
    args = parse_args()

    # default: e-value 1e-10 if -s not used alone
    evalue_thresh = args.evalue
    if evalue_thresh is None and not args.significant:
        evalue_thresh = 1e-10

    folder = args.folder
    if not os.path.isdir(folder):
        sys.exit(f"Error: '{folder}' is not a directory")

    # collect all files
    files = sorted(f for f in os.listdir(folder)
                   if os.path.isfile(os.path.join(folder, f)) and not f.startswith("."))
    if not files:
        sys.exit(f"Error: no files found in '{folder}'")

    # parse each file
    sample_names = []
    all_data = {}       # {sample: {KO: count}}
    all_desc = {}       # {KO: description}

    for fname in files:
        sample = os.path.splitext(fname)[0]
        sample_names.append(sample)
        ko_proteins, ko_desc = parse_file(
            os.path.join(folder, fname), evalue_thresh, args.significant
        )
        all_data[sample] = {ko: len(prots) for ko, prots in ko_proteins.items()}
        for ko, desc in ko_desc.items():
            if ko not in all_desc:
                all_desc[ko] = desc

    # all KOs found in at least one file
    all_kos = sorted(set().union(*(d.keys() for d in all_data.values())))

    # write output
    out = open(args.output, "w", newline="") if args.output else sys.stdout
    writer = csv.writer(out, delimiter="\t")
    writer.writerow(["KO"] + sample_names + ["ENTRY_AC", "ENTRY_TYPE", "ENTRY_NAME"])
    for ko in all_kos:
        row = [ko] + [all_data[s].get(ko, 0) for s in sample_names] + [ko, args.method, all_desc.get(ko, "")]
        writer.writerow(row)

    if args.output:
        out.close()
        print(f"Written to {args.output}", file=sys.stderr)

if __name__ == "__main__":
    main()
