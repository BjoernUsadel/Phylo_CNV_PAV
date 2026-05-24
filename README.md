# IPR PhyloSignal — Phylogenetic signal in InterPro domain repertoires

Detect InterPro domains whose presence/absence or copy-number variation correlates with phylogenetic structure across species or samples.

## Overview

This toolkit consists of two scripts:

1. **`summarize_ipr.py`** — Reads a folder of InterProScan TSV output files (one per sample), filters by e-value, and produces a count matrix: rows = IPR domains, columns = samples.

2. **`analyze_ipr_signal.py`** — Takes the count matrix and a phylogenetic tree, tests each domain for phylogenetic signal, and outputs ranked tables and publication-quality SVG heatmaps.

## Installation

```bash
pip install biopython numpy
```

Python 3.8+ required. Works on Linux, macOS, and Windows.

## Quick start

```bash
# Step 1: Build the IPR count matrix from InterProScan output
python summarize_ipr.py interproscan_results/ > ipr_counts.tsv

# Step 2: Test for phylogenetic signal (binary test only, fast)
python analyze_ipr_signal.py ipr_counts.tsv tree.newick -n 999 -o results/

# Step 2 (full): Binary + continuous tests, with species collapsing
python analyze_ipr_signal.py ipr_counts.tsv tree.newick \
    -m sample_to_species.tsv -n 999 -c -o results/
```

## Script 1: summarize_ipr.py

### What it does

Reads all `.tsv` files in a folder (one InterProScan output per sample), filters rows by e-value threshold, and counts the number of unique proteins per IPR domain per sample. Outputs a tab-separated matrix to stdout.

### Usage

```
python summarize_ipr.py [-e EVALUE] folder
```

| Argument | Description |
|----------|-------------|
| `folder` | Directory containing InterProScan `.tsv` files |
| `-e`, `--evalue` | E-value threshold (default: `1e-10`) |

### Filtering rules

- Rows without an IPR mapping (column 12 = `-`) are skipped.
- Rows where the e-value is `-` (e.g. Coils, MobiDBLite, SignalP, TMHMM) are excluded. See the code comment — some of these analysis types can carry meaningful IPR mappings; revisit if needed.
- Each unique protein counts once per IPR domain per file (deduplication).

### Output format

Tab-separated, to stdout:

```
IPR          sample_A   sample_B   sample_C
IPR000001    5          0          3
IPR000002    12         15         11
```

Column headers are filenames without the `.tsv` extension. The annotated variant (with `ENTRY_AC`, `ENTRY_TYPE`, `ENTRY_NAME` trailing columns) is also supported by `analyze_ipr_signal.py`.

### Example

```bash
python summarize_ipr.py -e 1e-20 interproscan_output/ > ipr_matrix.tsv
```

## Script 2: analyze_ipr_signal.py

### What it does

For each IPR domain, tests whether its distribution across samples follows the phylogeny more than expected by chance. Supports two complementary tests:

- **Binary (presence/absence):** Fitch parsimony + tip-label permutation test.
- **Continuous (copy-number):** Phylogenetic dispersion (inverse-distance-weighted) + permutation test.

### Usage

```
python analyze_ipr_signal.py [-m MAPPING] [-p PREFIX] [-n N_PERM]
                              [-c] [--top TOP] [--seed SEED] [-o OUTDIR]
                              matrix tree
```

| Argument | Description |
|----------|-------------|
| `matrix` | IPR count matrix TSV (output of `summarize_ipr.py`, optionally with annotation columns) |
| `tree` | Newick phylogeny (branch lengths used if present) |
| `-m`, `--mapping` | Optional sample-to-unit mapping TSV (`sample<TAB>unit`) |
| `-p`, `--prefix` | Prefix to strip from tree tip labels (default: `mycobiont_`) |
| `-n`, `--n-perm` | Number of permutations for p-values (default: `999`) |
| `-c`, `--continuous` | Also run the copy-number dispersion test (slower) |
| `--top` | Number of top domains in the heatmap (default: `100`) |
| `--seed` | Random seed for reproducibility (default: `42`) |
| `-o`, `--outdir` | Output directory (default: `.`) |

### Output files

Without `-c` (binary only):

| File | Content |
|------|---------|
| `signal_samples.tsv` | All domains ranked by binary p-value |
| `heatmap_samples_binary.svg` | Tree + heatmap, top 100 domains by parsimony signal |

With `-c` (binary + continuous):

| File | Content |
|------|---------|
| `signal_samples.tsv` | All domains, both p-values in one file |
| `heatmap_samples_binary.svg` | Ranked by binary (parsimony) p-value |
| `heatmap_samples_continuous.svg` | Ranked by continuous (dispersion) p-value |

With `-m` (unit-level collapsing), the same set is also produced for units:
`signal_units.tsv`, `heatmap_units_binary.svg`, `heatmap_units_continuous.svg`.

### Output TSV columns

| Column | Description |
|--------|-------------|
| `ipr` | InterPro accession |
| `name` | Human-readable domain name (if annotated input) |
| `n_present` | Number of tips where domain is present |
| `n_absent` | Number of tips where domain is absent |
| `parsimony` | Fitch parsimony score (minimum gains + losses) |
| `p_binary` | Permutation p-value for parsimony test |
| `enriched_clade` | Most enriched subtree description |
| `mean_count` | Mean copy number across tips |
| `max_count` | Maximum copy number |
| `cv` | Coefficient of variation |
| `p_continuous` | Permutation p-value for dispersion test (with `-c`) |
| `k_star` | Effect size K* ≈ Blomberg's K (with `-c`) |

### Sample-to-unit mapping

Optional two-column TSV, one line per sample:

```
F1_3    P. rufescens
F01     P. rufescens
F04     P. rufescens
F2      P. monticola
```

When provided, samples are collapsed to units:
- **Counts:** median across samples within each unit.
- **Presence:** a domain is "present" in a unit if ≥50% of its samples have it.

### Tree pruning

The tree may contain more tips than the matrix. Tips not matching any matrix sample (after prefix stripping) are pruned automatically. A custom recursive pruner is used instead of Bio.Phylo's built-in `prune()`, which can lose branch lengths during cascading unifurcation collapses. All pairwise distances between retained tips are preserved exactly.

## Methods (for publications)

### Binary signal (presence/absence)

Phylogenetic signal in domain presence/absence was assessed using Fitch parsimony (Fitch 1971). For each domain, the minimum number of evolutionary transitions (gains or losses) required to explain the observed distribution on the phylogeny was computed. Significance was evaluated by randomly permuting tip labels *N* times (Maddison & Slatkin 1991) and recording the fraction of permutations yielding parsimony scores ≤ the observed score, yielding a one-tailed p-value. To avoid p = 0, the formula p = (k + 1) / (N + 1) was used following North, Curtis & Sham (2002).

### Continuous signal (copy number)

Copy-number signal was assessed with a phylogenetic dispersion metric analogous to Moran's I. For each domain:

$$D = \frac{\sum_{i<j}\; w_{ij}\,(x_i - x_j)^2}{\sum_{i<j}\; w_{ij}}$$

where $x_i$ is the copy number at tip $i$ and $w_{ij} = 1/d_{ij}$ is the inverse phylogenetic distance between tips $i$ and $j$. Low $D$ indicates that closely related taxa share similar copy numbers (phylogenetic signal). An effect-size ratio $K^* = \text{median}(D_{\text{permuted}}) / D_{\text{observed}}$ is reported, analogous to Blomberg's K (Blomberg et al. 2003): $K^* > 1$ suggests stronger clustering by phylogeny than expected by chance. Significance is assessed by tip-label permutation.

### References

- Blomberg SP, Garland T, Ives AR (2003) Testing for phylogenetic signal in comparative data: behavioral traits are more labile. *Evolution* 57:717–745.
- Fitch WM (1971) Toward defining the course of evolution: minimum change for a specific tree topology. *Systematic Zoology* 20:406–416.
- Maddison WP, Slatkin M (1991) Null models for the number of evolutionary steps in a character on a phylogenetic tree. *Evolution* 45:1184–1197.
- North BV, Curtis D, Sham PC (2002) A note on the calculation of empirical P values from Monte Carlo procedures. *American Journal of Human Genetics* 71:439–441.

## Testing

```bash
python -m unittest test_analyze_ipr_signal -v
```

The test suite (24 tests) covers:
- Fitch parsimony correctness (invariant, clade, scattered, singleton patterns)
- Permutation test p-values (significant vs non-significant cases)
- Continuous dispersion test (clade-structured vs noisy counts)
- Clade enrichment detection
- Full pipeline integration (all domain types through `analyze()`)
- **Tree pruning distance preservation** (verifies Bio.Phylo branch-length bug is avoided)
- End-to-end file I/O (annotated matrix, prefix handling, SVG/TSV output, unit collapsing)

## Known limitations

- The continuous test uses inverse phylogenetic distance as weights. Tips with zero branch length between them get a large but finite weight (1e6).
- K* is capped at 999 for display. Values >> 1 all mean "strong signal"; extreme values mainly occur for binary 0/1 data where the continuous test is redundant with the binary test.
- The permutation test does not correct for multiple testing across domains. Apply Benjamini-Hochberg or similar correction to the output p-values for genome-wide interpretation.
- Unit-level collapsing uses one representative tip per unit for the tree. If a unit is not monophyletic, a warning is issued and the first tip (in traversal order) is used.


