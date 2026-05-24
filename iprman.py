#!/usr/bin/env python3
"""
Summarize InterProScan TSV files by IPR domain across species.

Reads all .tsv files in a folder, filters by e-value (optional), and produces a
TSV matrix: rows = IPR domains, columns = species (one per file).
Values = number of unique proteins with that IPR domain passing the threshold.

If no -e flag is given, ALL rows with an IPR accession are included regardless
of e-value (including rows where e-value is '-').

Usage:
    python3 iprman.py /path/to/folder
    python3 iprman.py -e 1e-20 /path/to/folder
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
        "-e", "--evalue", type=float, default=None,
        help="E-value threshold. If omitted, all rows with an IPR accession are kept."
    )
    return p.parse_args()


def process_file(filepath, evalue_threshold):
    """
    Returns:
        ipr_proteins : dict  {ipr_id: set of protein_ids}
        ipr_databases: dict  {ipr_id: set of member-database names}
        ipr_descs    : dict  {ipr_id: str}  (IPR description, last one wins)
    """
    ipr_proteins = defaultdict(set)
    ipr_databases = defaultdict(set)
    ipr_descs = {}
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
            member_db = cols[3]       # e.g. Pfam, PANTHER, CDD …
            evalue_raw = cols[8]
            ipr = cols[11]

            # Skip rows without IPR mapping
            if ipr == "-" or ipr == "":
                continue

            # --- e-value filtering ---
            if evalue_threshold is not None:
                # When a threshold is set we must be able to compare numerically,
                # so rows with a non-numeric e-value are skipped.
                if evalue_raw == "-" or evalue_raw == "":
                    continue
                try:
                    evalue = float(evalue_raw)
                except ValueError:
                    warnings.warn(
                        f"{basename}:{lineno}: cannot parse e-value '{evalue_raw}', skipping"
                    )
                    continue
                if evalue > evalue_threshold:
                    continue

            # Record hit
            ipr_proteins[ipr].add(protein)
            ipr_databases[ipr].add(member_db)

            # Grab IPR description (column 12) if present
            if len(cols) > 12 and cols[12] and cols[12] != "-":
                ipr_descs[ipr] = cols[12]

    return ipr_proteins, ipr_databases, ipr_descs


def main():
    args = parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"Error: '{args.folder}' is not a directory")

    files = sorted(glob.glob(os.path.join(args.folder, "*.tsv")))
    if not files:
        sys.exit(f"Error: no .tsv files found in '{args.folder}'")

    # Accumulators across all files
    all_samples = {}          # {sample: {ipr: count}}
    all_iprs = set()
    global_databases = defaultdict(set)   # {ipr: set of member dbs}
    global_descs = {}                     # {ipr: description}

    for filepath in files:
        sample = os.path.splitext(os.path.basename(filepath))[0]
        ipr_proteins, ipr_databases, ipr_descs = process_file(filepath, args.evalue)

        counts = {ipr: len(prots) for ipr, prots in ipr_proteins.items()}
        all_samples[sample] = counts
        all_iprs.update(counts.keys())

        for ipr, dbs in ipr_databases.items():
            global_databases[ipr].update(dbs)
        for ipr, desc in ipr_descs.items():
            global_descs[ipr] = desc          # last description wins

    if not all_iprs:
        warnings.warn("No IPR domains found after filtering")

    sample_names = sorted(all_samples.keys())
    iprs_sorted = sorted(all_iprs)

    # Header
    header = ["IPR"] + sample_names + ["ENTRY_AC", "ENTRY_TYPE", "ENTRY_NAME"]
    print("\t".join(header))

    # Rows
    for ipr in iprs_sorted:
        counts = [str(all_samples[s].get(ipr, 0)) for s in sample_names]
        entry_ac = ipr
        entry_type = ",".join(sorted(global_databases.get(ipr, set())))
        entry_name = global_descs.get(ipr, "")
        row = [ipr] + counts + [entry_ac, entry_type, entry_name]
        print("\t".join(row))


if __name__ == "__main__":
    main()
