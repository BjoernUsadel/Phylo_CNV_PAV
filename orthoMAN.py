#!/usr/bin/env python3
"""Summarize OrthoFinder orthogroups with optional KEGG and InterProScan annotations."""

import argparse
import csv
import os
import sys
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser(description="Summarize orthogroup gene counts with optional KEGG/IPR descriptions.")
    p.add_argument("orthogroups", help="Orthogroups.tsv file")
    p.add_argument("-o", "--method", default="orthofinder", help="ENTRY_TYPE value (default: orthofinder)")
    p.add_argument("-kegg", "--kegg_folder", default=None, help="Folder with KEGG scan files (one per sample)")
    p.add_argument("-kegg-e", "--kegg_evalue", type=float, default=1e-10, help="E-value threshold for KEGG (default: 1e-10)")
    p.add_argument("-ipr", "--ipr_folder", default=None, help="Folder with InterProScan files (one per sample)")
    p.add_argument("-ipr-e", "--ipr_evalue", type=float, default=1e-10, help="E-value threshold for InterProScan (default: 1e-10)")
    p.add_argument("-out", "--output", default=None, help="Output file (default: stdout)")
    return p.parse_args()

def parse_kegg_file(filepath, evalue_thresh):
    """Return {gene_name: set(descriptions)} for entries passing * AND e-value filter."""
    gene_descs = defaultdict(set)
    with open(filepath, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if parts[0].startswith("#"):
                continue
            if len(parts) < 7:
                continue

            sig = parts[0].strip()
            gene = parts[1].strip()
            ko = parts[2].strip()
            evalue_str = parts[5].strip()
            desc = parts[6].strip().strip('"')

            # must have * significance
            if sig != "*":
                continue
            # must pass e-value
            if not evalue_str:
                continue
            try:
                ev = float(evalue_str)
            except ValueError:
                continue
            if ev > evalue_thresh:
                continue

            if desc and ko:
                gene_descs[gene].add(f"{ko}: {desc}")
    return gene_descs

def parse_ipr_file(filepath, evalue_thresh):
    """Return {gene_name: set(descriptions)} for entries passing e-value filter."""
    gene_descs = defaultdict(set)
    with open(filepath, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 13:
                continue

            gene = parts[0].strip()
            evalue_str = parts[8].strip()
            ipr_acc = parts[11].strip()

            # skip entries with no e-value
            if evalue_str == "-" or not evalue_str:
                continue
            try:
                ev = float(evalue_str)
            except ValueError:
                continue
            if ev > evalue_thresh:
                continue

            # skip entries with no IPR accession
            if ipr_acc == "-" or not ipr_acc:
                continue

            desc = parts[12].strip() if parts[12].strip() not in ("-", "") else ""
            if desc:
                gene_descs[gene].add(f"{ipr_acc}: {desc}")
            else:
                gene_descs[gene].add(ipr_acc)
    return gene_descs

def find_sample_file(folder, sample_name):
    """Find a file matching the sample name in the folder."""
    # try exact match first, then with common extensions
    for candidate in [sample_name,
                      sample_name + ".txt", sample_name + ".tsv",
                      sample_name + ".tab"]:
        path = os.path.join(folder, candidate)
        if os.path.isfile(path):
            return path
    # try prefix match: filename starts with sample name (case-insensitive)
    matches = []
    for f in os.listdir(folder):
        if not os.path.isfile(os.path.join(folder, f)):
            continue
        base = os.path.splitext(f)[0]
        # exact match (case-insensitive)
        if base.lower() == sample_name.lower():
            return os.path.join(folder, f)
        # prefix match: file starts with sample name
        if f.lower().startswith(sample_name.lower()):
            matches.append(f)
    if matches:
        # pick shortest match to prefer closest name
        best = sorted(matches, key=len)[0]
        return os.path.join(folder, best)
    return None

def main():
    args = parse_args()

    # read orthogroups header to get sample names
    with open(args.orthogroups, errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        # strip \r from header fields
        header = [h.strip("\r") for h in header]
        sample_names = header[1:]  # first column is "Orthogroup"

        # pre-load KEGG and IPR annotations per sample
        kegg_annotations = {}  # {sample: {gene: set(descs)}}
        ipr_annotations = {}

        if args.kegg_folder:
            for sample in sample_names:
                fpath = find_sample_file(args.kegg_folder, sample)
                if fpath:
                    kegg_annotations[sample] = parse_kegg_file(fpath, args.kegg_evalue)
                else:
                    print(f"WARNING: no KEGG file found for sample '{sample}' in {args.kegg_folder}", file=sys.stderr)

        if args.ipr_folder:
            for sample in sample_names:
                fpath = find_sample_file(args.ipr_folder, sample)
                if fpath:
                    ipr_annotations[sample] = parse_ipr_file(fpath, args.ipr_evalue)
                else:
                    print(f"WARNING: no IPR file found for sample '{sample}' in {args.ipr_folder}", file=sys.stderr)

        # process orthogroups
        out = open(args.output, "w", newline="") if args.output else sys.stdout
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["Orthogroup"] + sample_names + ["ENTRY_AC", "ENTRY_TYPE", "ENTRY_NAME"])

        for row in reader:
            row = [c.strip("\r") for c in row]
            if not row or not row[0]:
                continue
            og_id = row[0]
            counts = []
            all_descs = set()

            for i, sample in enumerate(sample_names):
                cell = row[i + 1].strip() if i + 1 < len(row) else ""
                if not cell:
                    counts.append(0)
                    continue

                genes = [g.strip() for g in cell.split(",") if g.strip()]
                counts.append(len(genes))

                # collect descriptions for these genes
                for gene in genes:
                    if sample in kegg_annotations:
                        all_descs.update(kegg_annotations[sample].get(gene, set()))
                    if sample in ipr_annotations:
                        all_descs.update(ipr_annotations[sample].get(gene, set()))

            desc_str = "; ".join(sorted(all_descs)) if all_descs else ""
            writer.writerow([og_id] + counts + [og_id, args.method, desc_str])

    if args.output:
        out.close()
        print(f"Written to {args.output}", file=sys.stderr)

if __name__ == "__main__":
    main()
