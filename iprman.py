#!/usr/bin/env python3
"""
Summarize InterProScan TSV files by IPR domain across species.
 
Reads all .tsv files in a folder, filters by e-value, and produces a
TSV matrix: rows = IPR domains, columns = species (one per file).
Values = number of unique proteins with that IPR domain passing the threshold.
python3 summarize_ipr.py /path/to/folder or python3 summarize_ipr.py -e 1e-20 /path/to/folder with -e as e value threshold
"""
 
import argparse
import glob
import os
import sys
import warnings
from collections import defaultdict
 
def parse_args():
    p = argparse.ArgumentParser(
        description="Summarize InterProScan domains across species files."
    )
    p.add_argument("folder", help="Folder containing InterProScan .tsv files")
    p.add_argument(
        "-e", "--evalue", type=float, default=1e-10,
        help="E-value threshold (default: 1e-10)"
    )
    return p.parse_args()
 
 
def process_file(filepath, evalue_threshold):
    """Returns dict {ipr_id: set of protein_ids} for one file."""
    ipr_proteins = defaultdict(set)
    basename = os.path.basename(filepath)
 
    with open(filepath) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
 
            if len(cols) < 12:
                warnings.warn(f"{basename}:{lineno}: only {len(cols)} columns, skipping")
                continue
 
            protein = cols[0]
            evalue_raw = cols[8]
            ipr = cols[11]
 
            # Skip rows without IPR mapping
            if ipr == "-" or ipr == "":
                continue
 
            # NOTE: Rows where e-value is '-' (e.g. Coils, MobiDBLite, SignalP,
            # TMHMM, Phobius) are currently EXCLUDED. Some of these analysis types
            # can carry meaningful IPR mappings. Revisit this if needed.
            if evalue_raw == "-" or evalue_raw == "":
                continue
 
            try:
                evalue = float(evalue_raw)
            except ValueError:
                warnings.warn(f"{basename}:{lineno}: cannot parse e-value '{evalue_raw}', skipping")
                continue
 
            if evalue > evalue_threshold:
                continue
 
            ipr_proteins[ipr].add(protein)
 
    return ipr_proteins
 
 
def main():
    args = parse_args()
 
    if not os.path.isdir(args.folder):
        sys.exit(f"Error: '{args.folder}' is not a directory")
 
    files = sorted(glob.glob(os.path.join(args.folder, "*.tsv")))
    if not files:
        sys.exit(f"Error: no .tsv files found in '{args.folder}'")
 
    # {sample_name: {ipr: count}}
    all_samples = {}
    all_iprs = set()
 
    for filepath in files:
        sample = os.path.splitext(os.path.basename(filepath))[0]
        ipr_proteins = process_file(filepath, args.evalue)
        counts = {ipr: len(prots) for ipr, prots in ipr_proteins.items()}
        all_samples[sample] = counts
        all_iprs.update(counts.keys())
 
    if not all_iprs:
        warnings.warn("No IPR domains found after filtering")
 
    sample_names = sorted(all_samples.keys())
    iprs_sorted = sorted(all_iprs)
 
    # Write header
    print("\t".join(["IPR"] + sample_names))
 
    # Write rows
    for ipr in iprs_sorted:
        row = [ipr] + [str(all_samples[s].get(ipr, 0)) for s in sample_names]
        print("\t".join(row))
 
 
if __name__ == "__main__":
    main()
