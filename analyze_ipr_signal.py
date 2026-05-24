#!/usr/bin/env python3
"""
analyze_ipr_signal.py — Phylogenetic signal in InterPro domain repertoires
==========================================================================

Detects InterPro domains whose presence/absence or copy-number variation
correlates with phylogenetic structure (i.e. closely related taxa share
domain profiles more than expected by chance).

Methods (for use in publications)
---------------------------------
Presence/absence signal was assessed with Fitch parsimony (Fitch 1971).
For each domain, the minimum number of evolutionary transitions (gains or
losses) required to explain the observed distribution on the phylogeny was
computed.  Significance was evaluated by randomly permuting tip labels N
times (Maddison & Slatkin 1991, Syst. Zool. 40:315-328) and recording the
fraction of permuted parsimony scores <= the observed score, yielding a
one-tailed p-value.  To avoid p = 0, the observed score is included in
both numerator and denominator: p = (k + 1) / (N + 1), following
North, Curtis & Sham 2002 (Am. J. Hum. Genet. 71:439-441).

Copy-number signal (optional, --continuous) was assessed with a phylogenetic
dispersion metric analogous to Moran's I.  For each domain, we computed:

    D = sum_{i<j} w_ij * (x_i - x_j)^2  /  sum_{i<j} w_ij

where x_i is the copy number at tip i and w_ij = 1 / d_ij is the inverse
phylogenetic distance between tips i and j.  Low D indicates that closely
related taxa have similar copy numbers (phylogenetic signal).  An effect-
size ratio K* = median(D_permuted) / D_observed is reported, analogous to
Blomberg's K (Blomberg et al. 2003, Evolution 57:717-745): K* > 1 suggests
stronger clustering by phylogeny than expected by chance.  Significance is
again assessed by tip-label permutation.

Inputs
------
- IPR summary matrix TSV: first column = IPR_ID, then one column per
  sample with counts.  May optionally have trailing annotation columns
  (ENTRY_AC, ENTRY_TYPE, ENTRY_NAME); these are auto-detected.
- Newick phylogeny (branch lengths used if present).
- Optional sample-to-unit mapping TSV (sample<TAB>unit).

Outputs
-------
- signal_samples.tsv (and signal_units.tsv): ranked domain table.
- heatmap_samples.svg (and heatmap_units.svg): tree + heatmap.

Dependencies: biopython, numpy  (pip install biopython numpy)
"""

import argparse
import csv
import os
import sys
import random
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
from Bio import Phylo


# ===================================================================
# 1. COMMAND-LINE INTERFACE
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Test InterPro domains for phylogenetic signal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See script docstring for methods suitable for publication.",
    )
    p.add_argument("matrix",
        help="IPR matrix TSV (optionally with ENTRY_AC / ENTRY_TYPE / "
             "ENTRY_NAME trailing columns)")
    p.add_argument("tree", help="Newick tree file")
    p.add_argument("-m", "--mapping",
        help="Sample-to-unit mapping TSV (sample<TAB>unit), optional")
    p.add_argument("-p", "--prefix", default="mycobiont_",
        help="Prefix to strip from tree tip labels (default: mycobiont_)")
    p.add_argument("-n", "--n-perm", type=int, default=999,
        help="Number of permutations for p-values (default: 999)")
    p.add_argument("-c", "--continuous", action="store_true",
        help="Also run copy-number dispersion test (slower)")
    p.add_argument("--top", type=int, default=100,
        help="Number of top domains shown in heatmap (default: 100)")
    p.add_argument("--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)")
    p.add_argument("-o", "--outdir", default=".",
        help="Output directory (default: .)")
    return p.parse_args()


# ===================================================================
# 2. I/O — READING MATRIX, TREE, MAPPING
# ===================================================================

def read_matrix(path):
    """Read an IPR count matrix with optional annotation columns.

    Auto-detects trailing non-numeric columns (ENTRY_AC, ENTRY_TYPE,
    ENTRY_NAME) by checking whether the last column header is 'ENTRY_NAME'.

    Returns
    -------
    samples : list of str
        Column headers that are sample names.
    ipr_data : dict  {ipr_id: {sample: int}}
        Count matrix.
    ipr_names : dict  {ipr_id: str}
        Human-readable name per domain (empty string if absent).
    """
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = [h.strip() for h in next(reader)]

        # Detect annotation columns: expect ENTRY_NAME as last column
        has_annot = header[-1].upper() == "ENTRY_NAME"
        if has_annot:
            # Find where annotation starts: look for ENTRY_AC
            try:
                annot_start = next(
                    i for i, h in enumerate(header)
                    if h.upper() == "ENTRY_AC"
                )
            except StopIteration:
                # Fallback: last 3 columns
                annot_start = len(header) - 3
            samples = header[1:annot_start]
        else:
            samples = header[1:]
            annot_start = len(header)

        ipr_data = {}
        ipr_names = {}

        for row in reader:
            row = [c.strip() for c in row]
            ipr = row[0]
            try:
                counts = {
                    samples[i]: int(row[i + 1])
                    for i in range(len(samples))
                }
            except (ValueError, IndexError) as e:
                warnings.warn(f"Skipping malformed row for {ipr}: {e}")
                continue
            ipr_data[ipr] = counts

            # Store human-readable name if present
            if has_annot and len(row) > annot_start + 2:
                ipr_names[ipr] = row[annot_start + 2]  # ENTRY_NAME column
            else:
                ipr_names[ipr] = ""

    return samples, ipr_data, ipr_names


def read_mapping(path):
    """Read two-column TSV: sample<TAB>unit.  Lines starting with # are
    skipped.  Returns dict {sample: unit}."""
    mapping = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                warnings.warn(f"Mapping: skipping malformed line: {line!r}")
                continue
            mapping[parts[0]] = parts[1]
    return mapping


# ===================================================================
# 3. TREE — MATCHING, PRUNING, COLLAPSING
# ===================================================================

def _prune_to_tips(tree, keep_names):
    """Prune tree to only *keep_names* tips, preserving pairwise distances.

    Bio.Phylo.prune() can lose branch lengths during cascading
    unifurcation collapses.  This function avoids that by recursing
    top-down: at each internal node, children without keeper descendants
    are dropped; if only one child survives, the node is collapsed by
    summing its branch length into the surviving child's branch length.
    """
    tree = deepcopy(tree)
    keep = set(keep_names)

    def _prune(clade):
        if clade.is_terminal():
            return clade if clade.name in keep else None

        surviving = []
        for child in clade.clades:
            result = _prune(child)
            if result is not None:
                surviving.append(result)

        if not surviving:
            return None

        if len(surviving) == 1:
            # Unifurcation → collapse: merge branch lengths
            child = surviving[0]
            child.branch_length = (clade.branch_length or 0) + (child.branch_length or 0)
            return child

        clade.clades = surviving
        return clade

    new_root = _prune(tree.root)
    if new_root is None:
        raise ValueError("No matching tips found in tree")
    tree.root = new_root
    tree.root.branch_length = None  # root has no parent branch
    return tree


def match_and_prune(tree, matrix_samples, prefix):
    """Prune tree to tips whose label (after stripping *prefix*) appears
    in *matrix_samples*.  Renames surviving tips to the stripped name.

    Uses a custom recursive pruner that correctly preserves all pairwise
    distances (Bio.Phylo.prune() can lose branch lengths).

    Returns (pruned_tree, set_of_matched_sample_names).
    """
    # Build keep-set of original tip names
    tip_map = {}  # original_tip_name -> stripped_sample_name
    for tip in tree.get_terminals():
        stripped = tip.name
        if stripped and stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
        if stripped in matrix_samples:
            tip_map[tip.name] = stripped

    # Prune with correct branch-length handling
    tree = _prune_to_tips(tree, set(tip_map.keys()))

    # Rename remaining tips to sample names
    for tip in tree.get_terminals():
        tip.name = tip_map[tip.name]

    matched = {t.name for t in tree.get_terminals()}
    unmatched = matrix_samples - matched
    if unmatched:
        warnings.warn(
            f"{len(unmatched)} matrix sample(s) not found in tree: "
            f"{', '.join(sorted(unmatched)[:5])}"
            f"{'...' if len(unmatched) > 5 else ''}"
        )
    return tree, matched


def collapse_tree(tree, samples, mapping):
    """Collapse a sample-level tree to taxonomic-unit level.

    For each unit with multiple samples: keeps one representative tip
    (the first in tree traversal order), prunes the rest, and renames
    the kept tip to the unit name.  Warns if a unit is not monophyletic.

    Uses _prune_to_tips() to correctly preserve branch lengths.
    """
    tree = deepcopy(tree)
    unit_samples = defaultdict(list)
    for s in samples:
        if s in mapping:
            unit_samples[mapping[s]].append(s)

    # Check monophyly before pruning
    for unit, samps in unit_samples.items():
        tips_in_tree = [t for t in tree.get_terminals() if t.name in samps]
        if len(tips_in_tree) >= 2 and not tree.is_monophyletic(tips_in_tree):
            warnings.warn(
                f"Unit '{unit}' is NOT monophyletic — collapsing anyway "
                f"(using first tip as representative)"
            )

    # Select one representative per unit
    keep_tips = set()
    rep_map = {}   # representative_tip_name -> unit_name
    for unit, samps in unit_samples.items():
        tips_in_tree = [t.name for t in tree.get_terminals() if t.name in samps]
        if tips_in_tree:
            keep = tips_in_tree[0]
            keep_tips.add(keep)
            rep_map[keep] = unit

    # Prune with correct distance handling
    tree = _prune_to_tips(tree, keep_tips)

    # Rename representatives to unit names
    for tip in tree.get_terminals():
        if tip.name in rep_map:
            tip.name = rep_map[tip.name]

    return tree


def collapse_matrix(ipr_data, samples, mapping):
    """Collapse sample-level counts to unit-level.

    - Counts: median across samples in each unit (rounded to int).
    - A domain is considered 'present' in a unit if >= 50% of its
      samples have count > 0.

    Returns (unit_names_sorted, collapsed_ipr_data).
    """
    unit_samples = defaultdict(list)
    for s in samples:
        if s in mapping:
            unit_samples[mapping[s]].append(s)
    units = sorted(unit_samples.keys())

    collapsed = {}
    for ipr, counts in ipr_data.items():
        collapsed[ipr] = {}
        for unit in units:
            vals = [counts.get(s, 0) for s in unit_samples[unit]]
            collapsed[ipr][unit] = int(np.median(vals))
    return units, collapsed


# ===================================================================
# 4. BINARY SIGNAL TEST — FITCH PARSIMONY + PERMUTATION
# ===================================================================

def fitch_parsimony(tree, tip_states):
    """Fitch (1971) maximum-parsimony algorithm for binary characters.

    Parameters
    ----------
    tree : Bio.Phylo tree
    tip_states : dict {tip_name: 0 or 1}

    Returns
    -------
    int : minimum number of state changes (gains + losses) on the tree.

    Notes
    -----
    Post-order traversal.  Each leaf is assigned its observed state as a
    singleton set.  At each internal node the intersection of all child
    sets is computed; if empty, the union is taken and the score is
    incremented by 1.  This correctly handles polytomies for binary data
    (a polytomy with mixed states requires exactly one change regardless
    of the number of children).
    """
    node_sets = {}
    score = 0

    def _post(clade):
        nonlocal score
        if clade.is_terminal():
            node_sets[id(clade)] = {tip_states.get(clade.name, 0)}
            return

        for child in clade.clades:
            _post(child)

        child_sets = [node_sets[id(c)] for c in clade.clades]
        # Intersect all children
        isect = child_sets[0]
        for cs in child_sets[1:]:
            isect = isect & cs

        if isect:
            node_sets[id(clade)] = isect
        else:
            node_sets[id(clade)] = set().union(*child_sets)
            score += 1

    _post(tree.root)
    return score


def parsimony_permutation_test(tree, tip_states, observed, n_perm):
    """Tip-label permutation test for parsimony (Maddison & Slatkin 1991).

    Returns p-value = (k + 1) / (N + 1), where k is the number of
    permutations with parsimony score <= observed.  The +1 avoids p = 0
    (North et al. 2002).
    """
    tips = list(tip_states.keys())
    states = list(tip_states.values())
    k = 0
    for _ in range(n_perm):
        shuffled = dict(zip(tips, random.sample(states, len(states))))
        if fitch_parsimony(tree, shuffled) <= observed:
            k += 1
    return (k + 1) / (n_perm + 1)


# ===================================================================
# 5. CONTINUOUS SIGNAL TEST — PHYLOGENETIC DISPERSION + PERMUTATION
# ===================================================================

def build_distance_matrix(tree, tip_names):
    """Precompute pairwise phylogenetic distances and inverse-distance
    weights for all tips.

    Returns W (n x n weight matrix, w_ij = 1/d_ij, diagonal = 0).
    This is computed once and reused for all domains.
    """
    n = len(tip_names)
    W = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = tree.distance(tip_names[i], tip_names[j])
            if d > 0:
                w = 1.0 / d
            else:
                # Zero-length branch: assign large but finite weight
                w = 1e6
            W[i, j] = w
            W[j, i] = w

    return W


def phylo_dispersion(values_vec, W):
    """Compute phylogenetic dispersion D (inverse-distance-weighted mean
    squared difference between all tip pairs).

        D = sum_{i<j} w_ij * (x_i - x_j)^2  /  sum_{i<j} w_ij

    Low D = closely related taxa have similar values = phylogenetic signal.
    """
    diff_sq = np.subtract.outer(values_vec, values_vec) ** 2
    # Use upper triangle only (each pair counted once)
    mask = np.triu(np.ones_like(W, dtype=bool), k=1)
    D_num = np.sum(W[mask] * diff_sq[mask])
    D_den = np.sum(W[mask])
    return D_num / D_den if D_den > 0 else 0.0


def dispersion_permutation_test(values_vec, W, observed, n_perm):
    """Permutation test for phylogenetic dispersion.

    Returns (p_value, k_star) where:
    - p_value: fraction of permutations with D <= observed (one-tailed)
    - k_star: median(D_permuted) / D_observed, analogous to Blomberg's K
              (K* > 1 means more phylogenetic signal than random)
    """
    perm_scores = np.empty(n_perm)
    for i in range(n_perm):
        perm_scores[i] = phylo_dispersion(np.random.permutation(values_vec), W)

    # One-tailed: low D = signal
    k = np.sum(perm_scores <= observed)
    p_value = (k + 1) / (n_perm + 1)

    # Effect size — capped at 999 for readability (values >> 1 all mean
    # "strong signal"; this mainly happens with binary 0/1 data where
    # the continuous test is redundant with the binary test)
    median_perm = float(np.median(perm_scores))
    k_star = median_perm / observed if observed > 0 else 0.0
    k_star = min(k_star, 999.0)

    return p_value, round(k_star, 2)


# ===================================================================
# 6. CLADE ENRICHMENT — WHICH SUBTREE IS ENRICHED?
# ===================================================================

def clade_enrichment(tree, tip_states):
    """Identify the subtree most enriched for domain presence.

    For each internal node, computes (fraction present inside) minus
    (fraction present outside).  Returns a human-readable description
    of the best-scoring clade, or '-' if the domain is invariant.
    """
    all_tips = [t.name for t in tree.get_terminals()]
    n_total = len(all_tips)
    n_present = sum(tip_states.get(t, 0) for t in all_tips)

    if n_present == 0 or n_present == n_total:
        return "-"

    best_score = 0.0
    best_desc = "-"

    for clade in tree.get_nonterminals():
        ctips = [t.name for t in clade.get_terminals()]
        nc = len(ctips)
        if nc >= n_total or nc < 2:
            continue

        nc_pres = sum(tip_states.get(t, 0) for t in ctips)
        frac_in = nc_pres / nc
        n_out = n_total - nc
        frac_out = (n_present - nc_pres) / n_out if n_out > 0 else 0
        score = frac_in - frac_out

        if score > best_score:
            best_score = score
            tips_sorted = sorted(ctips)
            if nc <= 4:
                label = "+".join(tips_sorted)
            else:
                label = f"clade({nc}tips:{'+'.join(tips_sorted[:3])}...)"
            best_desc = (f"{label} "
                         f"[{nc_pres}/{nc} vs {n_present - nc_pres}/{n_out}]")

    return best_desc


# ===================================================================
# 7. MAIN ANALYSIS LOOP
# ===================================================================

def analyze(tree, names, ipr_data, ipr_names, n_perm, do_continuous=False,
            W=None):
    """Run parsimony (and optionally dispersion) tests for every domain.

    Parameters
    ----------
    tree : Bio.Phylo tree
    names : list of tip names in tree-traversal order
    ipr_data : {ipr: {sample: count}}
    ipr_names : {ipr: human_readable_name}
    n_perm : number of permutations
    do_continuous : if True, also run phylogenetic dispersion test
    W : precomputed weight matrix (required if do_continuous)

    Returns list of result dicts, sorted by binary p-value.
    """
    results = []
    n_iprs = len(ipr_data)

    for i, (ipr, counts) in enumerate(ipr_data.items()):
        if (i + 1) % 200 == 0 or (i + 1) == n_iprs:
            print(f"  {i + 1}/{n_iprs}...", file=sys.stderr)

        vals = [counts.get(s, 0) for s in names]
        tip_states = {s: (1 if v > 0 else 0) for s, v in zip(names, vals)}
        n_pres = sum(tip_states.values())
        n_abs = len(names) - n_pres
        mean_c = round(float(np.mean(vals)), 1)
        max_c = int(max(vals))
        cv = (round(float(np.std(vals) / np.mean(vals)), 2)
              if np.mean(vals) > 0 else 0.0)
        name_str = ipr_names.get(ipr, "")

        row = dict(
            ipr=ipr, name=name_str, n_present=n_pres, n_absent=n_abs,
            parsimony=0, p_binary=1.0, enriched_clade="-",
            mean_count=mean_c, max_count=max_c, cv=cv,
            p_continuous=np.nan, k_star=np.nan,
        )

        # Invariant presence/absence: skip binary test but still check
        # continuous (counts can vary even if domain is always present)
        if n_pres > 0 and n_abs > 0:
            # --- Binary test ---
            pars = fitch_parsimony(tree, tip_states)
            p_bin = parsimony_permutation_test(tree, tip_states, pars, n_perm)
            enriched = clade_enrichment(tree, tip_states)
            row["parsimony"] = pars
            row["p_binary"] = p_bin
            row["enriched_clade"] = enriched

        # --- Continuous test (optional, needs count variation) ---
        if do_continuous and W is not None and np.var(vals) > 0:
            vals_arr = np.array(vals, dtype=float)
            D_obs = phylo_dispersion(vals_arr, W)
            p_cont, k_star = dispersion_permutation_test(
                vals_arr, W, D_obs, n_perm
            )
            row["p_continuous"] = p_cont
            row["k_star"] = k_star

        results.append(row)

    # Sort: binary p-value first, then parsimony as tiebreaker
    results.sort(key=lambda r: (r["p_binary"], r["parsimony"]))
    return results


# ===================================================================
# 8. OUTPUT — TSV
# ===================================================================

def write_results(results, path, include_continuous=False):
    """Write ranked results to TSV."""
    cols = [
        "ipr", "name", "n_present", "n_absent", "parsimony", "p_binary",
        "enriched_clade", "mean_count", "max_count", "cv",
    ]
    if include_continuous:
        cols += ["p_continuous", "k_star"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in results:
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, float) and np.isnan(v):
                    vals.append("")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")


# ===================================================================
# 9. OUTPUT — SVG HEATMAP
# ===================================================================

def _viridis(t):
    """Attempt a viridis-ish color gradient for t in [0, 1].
    0 = dark purple, 0.5 = teal, 1 = yellow."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        s = t / 0.5
        r = int(68 * (1 - s) + 33 * s)
        g = int(1 * (1 - s) + 145 * s)
        b = int(84 * (1 - s) + 140 * s)
    else:
        s = (t - 0.5) / 0.5
        r = int(33 * (1 - s) + 253 * s)
        g = int(145 * (1 - s) + 231 * s)
        b = int(140 * (1 - s) + 37 * s)
    return f"rgb({r},{g},{b})"


def _truncate(text, maxlen=40):
    """Shorten a label for SVG display."""
    return text if len(text) <= maxlen else text[:maxlen - 2] + ".."


def render_svg(tree, names_order, ipr_data, ipr_names, results, path,
               top_n, rank_by="binary"):
    """Render a publication-quality phylo-heatmap as SVG.

    Layout:  tree | tip labels | separator | heatmap columns
    Column headers show "IPR_ID: human name" where available.
    A thin colour bar under each header encodes p-value significance.

    rank_by : "binary" or "continuous"
        Controls which p-value is used to rank columns, colour the
        significance bar, and highlight column headers.
    """
    p_key = "p_binary" if rank_by == "binary" else "p_continuous"
    rank_label = "parsimony" if rank_by == "binary" else "dispersion"

    # Re-sort results by the chosen test, filter to variable + testable
    def _sortkey(r):
        pv = r[p_key]
        if isinstance(pv, float) and np.isnan(pv):
            return (2.0, 0)  # push untested domains to end
        return (pv, r["parsimony"])

    ranked = sorted(results, key=_sortkey)
    shown = [r for r in ranked
             if r["n_present"] > 0 and r["n_absent"] > 0
             and not (isinstance(r[p_key], float) and np.isnan(r[p_key]))
             ][:top_n]
    if not shown:
        warnings.warn("No variable domains to plot")
        return
    ipr_ids = [r["ipr"] for r in shown]

    n_tips = len(names_order)
    n_cols = len(ipr_ids)

    # Column display labels
    col_labels = []
    for r in shown:
        nm = (r.get("name") or "").strip()
        if nm:
            col_labels.append(f"{r['ipr']}: {_truncate(nm)}".replace("&", "&amp;"))
        else:
            col_labels.append(r["ipr"])

    # --- Layout geometry ---
    row_h = 18
    tree_w = 280
    gap1, gap2 = 8, 6
    label_w = max(len(n) for n in names_order) * 6.2 + 8
    cell_w = max(12, min(20, 1000 // max(n_cols, 1)))
    header_h = max(len(c) for c in col_labels) * 4.2 + 35
    title_h = 32
    ml, mr, mt, mb = 30, 30, 20, 70

    heat_x0 = ml + tree_w + gap1 + label_w + gap2
    total_w = heat_x0 + n_cols * cell_w + mr
    grid_top = mt + title_h + header_h
    total_h = grid_top + n_tips * row_h + mb

    # Tip y-positions
    tip_y = {name: grid_top + i * row_h + row_h / 2
             for i, name in enumerate(names_order)}

    # Tree depth coordinates
    real_depths = tree.depths()
    max_d = max(real_depths.values())
    if max_d == 0:
        real_depths = tree.depths(unit_branch_lengths=True)
        max_d = max(real_depths.values()) or 1

    def nx(clade):
        return ml + (real_depths[clade] / max_d) * tree_w

    def ny(clade):
        if clade.is_terminal():
            return tip_y.get(clade.name, 0)
        cys = [ny(c) for c in clade.clades]
        return (min(cys) + max(cys)) / 2

    S = []
    S.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'width="{total_w}" height="{total_h}" '
             f'viewBox="0 0 {total_w} {total_h}">')
    S.append("""<defs><style type="text/css">
  .title  { font-family: Arial, Helvetica, sans-serif; font-weight: 600;
             font-size: 14px; fill: #1a1a2e; }
  .sub    { font-family: Arial, sans-serif; font-size: 10px; fill: #666; }
  .tip    { font-family: Consolas, 'Courier New', monospace; font-size: 9px;
             fill: #2d2d2d; }
  .colhdr { font-family: Consolas, monospace; font-size: 6.5px; }
  .sig    { fill: #c0392b; font-weight: bold; }
  .ns     { fill: #555; }
  .cell   { font-family: Consolas, monospace; font-size: 7px;
             text-anchor: middle; dominant-baseline: central; }
  .tree   { stroke: #2d2d2d; stroke-width: 1.0; stroke-linecap: round; }
  .leg    { font-family: Arial, sans-serif; font-size: 9px; fill: #444; }
  .legb   { font-family: Arial, sans-serif; font-size: 9px; fill: #222;
             font-weight: 600; }
  .sc     { font-family: Arial, sans-serif; font-size: 8px; fill: #666; }
</style></defs>""")

    S.append(f'<rect width="{total_w}" height="{total_h}" fill="white"/>')

    # Title
    n_sig = sum(1 for r in shown if r[p_key] < 0.05)
    S.append(f'<text x="{ml}" y="{mt + 14}" class="title">'
             f'Phylogenetic signal — IPR domain heatmap '
             f'(ranked by {rank_label})</text>')
    S.append(f'<text x="{ml}" y="{mt + 27}" class="sub">'
             f'Top {len(shown)} variable domains ({n_sig} with '
             f'p_{rank_by} &lt; 0.05) | {n_tips} tips</text>')

    # Alternating row bands
    for i in range(n_tips):
        if i % 2 == 0:
            S.append(f'<rect x="0" y="{grid_top + i * row_h:.0f}" '
                     f'width="{total_w}" height="{row_h}" '
                     f'fill="#f8f9fa" opacity="0.5"/>')

    # --- Tree ---
    def draw(clade):
        x, y = nx(clade), ny(clade)
        if not clade.is_terminal():
            cys = [ny(c) for c in clade.clades]
            S.append(f'<line x1="{x:.1f}" y1="{min(cys):.1f}" '
                     f'x2="{x:.1f}" y2="{max(cys):.1f}" class="tree"/>')
            for child in clade.clades:
                cy, cx = ny(child), nx(child)
                S.append(f'<line x1="{x:.1f}" y1="{cy:.1f}" '
                         f'x2="{cx:.1f}" y2="{cy:.1f}" class="tree"/>')
                draw(child)
    draw(tree.root)

    # Scale bar
    sbar_y = grid_top + n_tips * row_h + 14
    raw = max_d / 5
    if raw > 0:
        mag = 10 ** int(np.floor(np.log10(raw)))
        nice = round(raw / mag) * mag or mag
        bpx = (nice / max_d) * tree_w
        S.append(f'<line x1="{ml}" y1="{sbar_y}" '
                 f'x2="{ml + bpx:.1f}" y2="{sbar_y}" '
                 f'stroke="#333" stroke-width="1.5"/>')
        S.append(f'<text x="{ml + bpx / 2:.1f}" y="{sbar_y + 11}" '
                 f'text-anchor="middle" class="sc">{nice:g}</text>')

    # Tip labels
    lx = ml + tree_w + gap1
    for name in names_order:
        S.append(f'<text x="{lx}" y="{tip_y[name] + 3:.1f}" '
                 f'class="tip">{name}</text>')

    # Separator
    sx = heat_x0 - 3
    S.append(f'<line x1="{sx}" y1="{grid_top - 2}" x2="{sx}" '
             f'y2="{grid_top + n_tips * row_h}" '
             f'stroke="#ccc" stroke-width="0.5"/>')

    # Column headers (rotated)
    for j, label in enumerate(col_labels):
        x = heat_x0 + j * cell_w + cell_w / 2
        y = grid_top - 5
        cls = "colhdr sig" if shown[j][p_key] < 0.05 else "colhdr ns"
        S.append(f'<text x="{x:.1f}" y="{y:.1f}" '
                 f'transform="rotate(-60 {x:.1f} {y:.1f})" '
                 f'class="{cls}">{label}</text>')

    # P-value colour bar
    pb_y = grid_top - 2
    for j, r in enumerate(shown):
        pv = r[p_key]
        c = ("#c0392b" if pv < 0.001 else
             "#e74c3c" if pv < 0.01  else
             "#f39c12" if pv < 0.05  else "#ecf0f1")
        S.append(f'<rect x="{heat_x0 + j * cell_w:.1f}" y="{pb_y:.1f}" '
                 f'width="{cell_w - 0.5}" height="3" fill="{c}" rx="0.5"/>')

    # Heatmap cells (log colour scale)
    all_v = [ipr_data[ipr].get(s, 0) for ipr in ipr_ids for s in names_order]
    max_val = max(all_v) or 1
    log_max = np.log1p(max_val)

    for j, ipr in enumerate(ipr_ids):
        cx0 = heat_x0 + j * cell_w
        for i, name in enumerate(names_order):
            val = ipr_data[ipr].get(name, 0)
            cy0 = grid_top + i * row_h
            if val == 0:
                fill = "#eef0f2"
                t = 0
            else:
                t = np.log1p(val) / log_max
                fill = _viridis(t)
            S.append(f'<rect x="{cx0 + 0.5:.1f}" y="{cy0 + 0.5:.1f}" '
                     f'width="{cell_w - 1:.1f}" height="{row_h - 1:.1f}" '
                     f'fill="{fill}" rx="1"/>')
            if val > 0 and cell_w >= 14:
                tf = "#fff" if t > 0.3 else "#222"
                S.append(f'<text x="{cx0 + cell_w / 2:.1f}" '
                         f'y="{cy0 + row_h / 2:.1f}" '
                         f'class="cell" fill="{tf}">{val}</text>')

    # --- Legends ---
    lg_y = grid_top + n_tips * row_h + 36
    lg_x = heat_x0
    bw = min(200, n_cols * cell_w)

    # Colour bar
    S.append(f'<text x="{lg_x}" y="{lg_y - 6}" class="legb">Count</text>')
    nst = 60
    for si in range(nst):
        t = si / (nst - 1)
        S.append(f'<rect x="{lg_x + si * bw / nst:.1f}" y="{lg_y}" '
                 f'width="{bw / nst + 0.5:.1f}" height="10" '
                 f'fill="{_viridis(t)}"/>')
    S.append(f'<text x="{lg_x}" y="{lg_y + 22}" class="sc">1</text>')
    mid = int(np.expm1(log_max / 2))
    S.append(f'<text x="{lg_x + bw / 2:.1f}" y="{lg_y + 22}" '
             f'text-anchor="middle" class="sc">{mid}</text>')
    S.append(f'<text x="{lg_x + bw:.1f}" y="{lg_y + 22}" '
             f'text-anchor="end" class="sc">{max_val}</text>')

    # Absent box
    ax = lg_x + bw + 20
    S.append(f'<rect x="{ax}" y="{lg_y}" width="10" height="10" '
             f'fill="#eef0f2" stroke="#ccc" stroke-width="0.5" rx="1"/>')
    S.append(f'<text x="{ax + 14}" y="{lg_y + 9}" class="leg">absent</text>')

    # P-value boxes
    px = ax + 65
    S.append(f'<text x="{px}" y="{lg_y - 6}" class="legb">'
             f'P-value ({rank_by})</text>')
    for lbl, col in [("&lt; 0.001", "#c0392b"), ("&lt; 0.01", "#e74c3c"),
                      ("&lt; 0.05", "#f39c12"), ("≥ 0.05", "#ecf0f1")]:
        S.append(f'<rect x="{px}" y="{lg_y}" width="10" height="10" '
                 f'fill="{col}" rx="1"/>')
        S.append(f'<text x="{px + 14}" y="{lg_y + 9}" class="leg">'
                 f'{lbl}</text>')
        px += 60

    S.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(S))


# ===================================================================
# 10. MAIN
# ===================================================================

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # --- Read inputs ---
    print("Reading matrix...", file=sys.stderr)
    samples, ipr_data, ipr_names = read_matrix(args.matrix)
    print(f"  {len(samples)} samples, {len(ipr_data)} IPR domains",
          file=sys.stderr)
    n_named = sum(1 for v in ipr_names.values() if v)
    if n_named:
        print(f"  {n_named} domains with human-readable names",
              file=sys.stderr)

    print("Reading tree...", file=sys.stderr)
    tree = Phylo.read(args.tree, "newick")

    print("Pruning tree to matrix samples...", file=sys.stderr)
    tree, matched = match_and_prune(tree, set(samples), args.prefix)
    print(f"  {len(matched)} samples matched", file=sys.stderr)
    if len(matched) < 3:
        sys.exit("Error: fewer than 3 samples matched between tree and matrix")

    # --- Precompute distance matrix for continuous test ---
    tip_order = [t.name for t in tree.get_terminals()]
    W = None
    if args.continuous:
        print("Precomputing phylogenetic distance matrix...", file=sys.stderr)
        W = build_distance_matrix(tree, tip_order)

    # --- Sample-level analysis ---
    print(f"Sample-level analysis ({args.n_perm} permutations, "
          f"continuous={'yes' if args.continuous else 'no'})...",
          file=sys.stderr)
    results_s = analyze(
        tree, tip_order, ipr_data, ipr_names, args.n_perm,
        do_continuous=args.continuous, W=W,
    )

    tsv_s = os.path.join(args.outdir, "signal_samples.tsv")
    write_results(results_s, tsv_s, include_continuous=args.continuous)
    print(f"  -> {tsv_s}", file=sys.stderr)

    svg_s = os.path.join(args.outdir, "heatmap_samples_binary.svg")
    render_svg(tree, tip_order, ipr_data, ipr_names, results_s, svg_s,
               args.top, rank_by="binary")
    print(f"  -> {svg_s}", file=sys.stderr)

    if args.continuous:
        svg_sc = os.path.join(args.outdir, "heatmap_samples_continuous.svg")
        render_svg(tree, tip_order, ipr_data, ipr_names, results_s, svg_sc,
                   args.top, rank_by="continuous")
        print(f"  -> {svg_sc}", file=sys.stderr)

    # --- Unit-level analysis (optional) ---
    if args.mapping:
        print("Reading mapping...", file=sys.stderr)
        mapping = read_mapping(args.mapping)

        unmapped = [s for s in matched if s not in mapping]
        if unmapped:
            warnings.warn(
                f"{len(unmapped)} matched sample(s) not in mapping, dropped: "
                f"{', '.join(sorted(unmapped)[:5])}"
            )

        mapped_samples = [s for s in samples if s in matched and s in mapping]
        units, collapsed = collapse_matrix(ipr_data, mapped_samples, mapping)
        print(f"  {len(units)} taxonomic units", file=sys.stderr)

        unit_tree = collapse_tree(tree, mapped_samples, mapping)
        unit_order = [t.name for t in unit_tree.get_terminals()]

        W_u = None
        if args.continuous:
            W_u = build_distance_matrix(unit_tree, unit_order)

        print(f"Unit-level analysis ({args.n_perm} permutations)...",
              file=sys.stderr)
        results_u = analyze(
            unit_tree, unit_order, collapsed, ipr_names, args.n_perm,
            do_continuous=args.continuous, W=W_u,
        )

        tsv_u = os.path.join(args.outdir, "signal_units.tsv")
        write_results(results_u, tsv_u, include_continuous=args.continuous)
        print(f"  -> {tsv_u}", file=sys.stderr)

        svg_u = os.path.join(args.outdir, "heatmap_units_binary.svg")
        render_svg(unit_tree, unit_order, collapsed, ipr_names, results_u,
                   svg_u, args.top, rank_by="binary")
        print(f"  -> {svg_u}", file=sys.stderr)

        if args.continuous:
            svg_uc = os.path.join(args.outdir, "heatmap_units_continuous.svg")
            render_svg(unit_tree, unit_order, collapsed, ipr_names, results_u,
                       svg_uc, args.top, rank_by="continuous")
            print(f"  -> {svg_uc}", file=sys.stderr)

    # --- Summary ---
    n_sig = sum(1 for r in results_s if r["p_binary"] < 0.05)
    print(f"\nDone. {n_sig}/{len(results_s)} domains with p_binary < 0.05 "
          f"(sample-level).", file=sys.stderr)
    if args.continuous:
        n_sig_c = sum(1 for r in results_s
                      if not np.isnan(r["p_continuous"])
                      and r["p_continuous"] < 0.05)
        print(f"      {n_sig_c}/{len(results_s)} domains with "
              f"p_continuous < 0.05.", file=sys.stderr)


if __name__ == "__main__":
    main()
