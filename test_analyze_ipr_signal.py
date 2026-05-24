#!/usr/bin/env python3
"""
Unit tests for analyze_ipr_signal.py

Uses a synthetic 10-tip tree with two clear clades and engineered IPR
domains that should produce specific outcomes:

    Tree topology:
                        ┌── s1
                   ┌────┤
                   │    └── s2
              ┌────┤
              │    │    ┌── s3
              │    └────┤
         ┌────┤         └── s4
         │    │
         │    └──────── s5      (long branch outgroup to left clade)
    root─┤
         │         ┌── s6
         │    ┌────┤
         │    │    └── s7
         └────┤
              │    ┌── s8
              └────┤
                   └── s9

    Left clade:  s1, s2, s3, s4, s5
    Right clade: s6, s7, s8, s9

    (Plus s10 as a distant outgroup to both clades.)

Test domains:
    IPR_INVARIANT_1    all present at count 1         → not testable
    IPR_INVARIANT_0    all absent                     → not testable
    IPR_CLADE_LEFT     left=1, right=0                → strong binary
    IPR_CLADE_RIGHT    left=0, right=1                → strong binary
    IPR_SCATTERED      alternating 1,0 across tree    → no binary signal
    IPR_SINGLETON      present in 1 tip only          → p=1 (1 change
                                                         is also expected)
    IPR_COUNTS_CLADE   all present, left=high right=1 → strong continuous
    IPR_COUNTS_NOISY   all present, random counts     → no continuous signal
    IPR_BINARY_ONLY    left=present(varied), right=0  → strong binary,
                                                         weak continuous
"""

import os
import sys
import unittest
import tempfile
import shutil
import random

import numpy as np
from io import StringIO
from Bio import Phylo

# Import the module under test
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_ipr_signal as sig


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

TEST_NEWICK = (
    "((((s1:0.1,s2:0.1):0.3,(s3:0.1,s4:0.1):0.3):0.2,"
    "s5:0.6):0.4,"
    "((s6:0.1,s7:0.1):0.3,(s8:0.1,s9:0.1):0.3):0.6,"
    "s10:1.0);"
)
# Tips in left clade:  s1, s2, s3, s4, s5
# Tips in right clade: s6, s7, s8, s9
# Outgroup: s10

TIPS = [f"s{i}" for i in range(1, 11)]
LEFT  = ["s1", "s2", "s3", "s4", "s5"]
RIGHT = ["s6", "s7", "s8", "s9"]

N_PERM = 499  # enough for reliable p-values in tests

# Shared temp file for the tree (avoids StringIO issues with Bio.Phylo)
_TREE_FILE = os.path.join(tempfile.gettempdir(), "_ipr_test_tree.nwk")


def _make_tree():
    with open(_TREE_FILE, "w") as f:
        f.write(TEST_NEWICK)
    return Phylo.read(_TREE_FILE, "newick")


def _make_counts(tip_values):
    """Helper: dict {tip: count} from a list of 10 values."""
    return dict(zip(TIPS, tip_values))


# Pre-built domain definitions: (ipr_id, counts_list, human_name)
DOMAINS = {
    # --- Invariant (not testable) ---
    "IPR_INVARIANT_1": ([1]*10,    "All-ones control"),
    "IPR_INVARIANT_0": ([0]*10,    "All-zeros control"),

    # --- Strong binary signal ---
    # Left clade present, right + outgroup absent → 1 change
    "IPR_CLADE_LEFT":  ([1,1,1,1,1, 0,0,0,0, 0], "Left-clade marker"),
    # Right clade present, left + outgroup absent → 1 change
    "IPR_CLADE_RIGHT": ([0,0,0,0,0, 1,1,1,1, 0], "Right-clade marker"),

    # --- No binary signal (scattered) ---
    # Alternating across tree order → many changes needed
    "IPR_SCATTERED":   ([1,0,1,0,1, 0,1,0,1, 0], "Scattered control"),

    # --- Singleton (present in 1 tip) ---
    "IPR_SINGLETON":   ([0,0,0,0,0, 0,0,0,1, 0], "Single-tip domain"),

    # --- Strong continuous signal (all present, counts differ by clade) ---
    # Left clade: high counts; right clade + outgroup: low counts
    "IPR_COUNTS_CLADE": ([8,9,7,10,8, 1,1,2,1, 1],
                         "Expansion in left clade"),

    # --- No continuous signal (all present, similar counts) ---
    "IPR_COUNTS_NOISY": ([3,1,4,2,5, 4,2,3,1, 5],
                         "Uniform-ish counts"),

    # --- Strong binary, weak continuous ---
    # Present in left clade only, but counts within left are chaotic
    # (5,1,1,8,2) — sub-clades (s1,s2) vs (s3,s4) aren't consistent
    "IPR_BINARY_ONLY": ([5,1,1,8,2, 0,0,0,0, 0],
                        "Binary signal, noisy counts"),
}


def _build_ipr_data():
    return {ipr: _make_counts(vals) for ipr, (vals, _) in DOMAINS.items()}


def _build_ipr_names():
    return {ipr: name for ipr, (_, name) in DOMAINS.items()}


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestFitchParsimony(unittest.TestCase):
    """Verify the Fitch algorithm gives correct parsimony scores."""

    def setUp(self):
        self.tree = _make_tree()

    def test_invariant_all_present(self):
        states = {t: 1 for t in TIPS}
        self.assertEqual(sig.fitch_parsimony(self.tree, states), 0)

    def test_invariant_all_absent(self):
        states = {t: 0 for t in TIPS}
        self.assertEqual(sig.fitch_parsimony(self.tree, states), 0)

    def test_perfect_clade_one_change(self):
        """Left clade = 1, right + outgroup = 0 → exactly 1 transition."""
        states = {t: (1 if t in LEFT else 0) for t in TIPS}
        self.assertEqual(sig.fitch_parsimony(self.tree, states), 1)

    def test_scattered_many_changes(self):
        """Alternating pattern requires many changes."""
        states = _make_counts([1,0,1,0,1, 0,1,0,1, 0])
        # Convert to binary
        states = {k: (1 if v > 0 else 0) for k, v in states.items()}
        score = sig.fitch_parsimony(self.tree, states)
        # Should need at least 3 changes for this scattered pattern
        self.assertGreaterEqual(score, 3)

    def test_singleton(self):
        """Single tip present → exactly 1 change."""
        states = {t: 0 for t in TIPS}
        states["s9"] = 1
        self.assertEqual(sig.fitch_parsimony(self.tree, states), 1)


class TestBinaryPermutation(unittest.TestCase):
    """Verify permutation test returns sensible p-values."""

    def setUp(self):
        random.seed(42)
        np.random.seed(42)
        self.tree = _make_tree()

    def test_clade_signal_significant(self):
        """Perfect clade pattern should be significant (p < 0.05)."""
        states = {t: (1 if t in LEFT else 0) for t in TIPS}
        pars = sig.fitch_parsimony(self.tree, states)
        p = sig.parsimony_permutation_test(self.tree, states, pars, N_PERM)
        self.assertLess(p, 0.05,
                        f"Expected significant p for clade pattern, got {p}")

    def test_scattered_not_significant(self):
        """Scattered pattern should NOT be significant (p > 0.1)."""
        states = _make_counts([1,0,1,0,1, 0,1,0,1, 0])
        states = {k: (1 if v > 0 else 0) for k, v in states.items()}
        pars = sig.fitch_parsimony(self.tree, states)
        p = sig.parsimony_permutation_test(self.tree, states, pars, N_PERM)
        self.assertGreater(p, 0.1,
                           f"Scattered pattern should not be significant, "
                           f"got p={p}")

    def test_invariant_returns_1(self):
        """Invariant domain → p = 1.0 (handled before permutation)."""
        # In the full pipeline invariant domains get p=1.0 without
        # running the permutation test, but verify parsimony = 0
        states = {t: 1 for t in TIPS}
        self.assertEqual(sig.fitch_parsimony(self.tree, states), 0)


class TestContinuousSignal(unittest.TestCase):
    """Verify phylogenetic dispersion and its permutation test."""

    def setUp(self):
        random.seed(42)
        np.random.seed(42)
        self.tree = _make_tree()
        self.W = sig.build_distance_matrix(self.tree, TIPS)

    def test_distance_matrix_shape(self):
        self.assertEqual(self.W.shape, (10, 10))
        # Diagonal should be 0
        np.testing.assert_array_equal(np.diag(self.W), 0)
        # Should be symmetric
        np.testing.assert_array_almost_equal(self.W, self.W.T)

    def test_counts_clade_significant(self):
        """High counts in left clade, low in right → strong continuous
        signal (low dispersion, low p)."""
        vals = np.array([8, 9, 7, 10, 8, 1, 1, 2, 1, 1], dtype=float)
        D = sig.phylo_dispersion(vals, self.W)
        p, k_star = sig.dispersion_permutation_test(vals, self.W, D, N_PERM)
        self.assertLess(p, 0.05,
                        f"Clade-structured counts should be significant, "
                        f"got p={p}")
        self.assertGreater(k_star, 1.0,
                           f"K* should be > 1 for phylo-structured counts, "
                           f"got {k_star}")

    def test_noisy_counts_not_significant(self):
        """Random-ish counts across all tips → no signal."""
        vals = np.array([3, 1, 4, 2, 5, 4, 2, 3, 1, 5], dtype=float)
        D = sig.phylo_dispersion(vals, self.W)
        p, k_star = sig.dispersion_permutation_test(vals, self.W, D, N_PERM)
        self.assertGreater(p, 0.1,
                           f"Noisy counts should not be significant, "
                           f"got p={p}")

    def test_zero_variance_skipped(self):
        """Constant counts → variance=0, dispersion=0."""
        vals = np.array([5]*10, dtype=float)
        D = sig.phylo_dispersion(vals, self.W)
        self.assertAlmostEqual(D, 0.0)


class TestCladeEnrichment(unittest.TestCase):
    """Verify clade enrichment detection."""

    def setUp(self):
        self.tree = _make_tree()

    def test_left_clade_detected(self):
        states = {t: (1 if t in LEFT else 0) for t in TIPS}
        desc = sig.clade_enrichment(self.tree, states)
        # Should mention the left-clade tips
        for tip in ["s1", "s2"]:
            self.assertIn(tip, desc,
                          f"Expected {tip} in enrichment description: {desc}")

    def test_invariant_returns_dash(self):
        states = {t: 1 for t in TIPS}
        self.assertEqual(sig.clade_enrichment(self.tree, states), "-")


class TestFullAnalyze(unittest.TestCase):
    """Integration test: run the full analyze() pipeline on synthetic data
    and verify each domain behaves as expected."""

    def setUp(self):
        random.seed(42)
        np.random.seed(42)
        self.tree = _make_tree()
        self.ipr_data = _build_ipr_data()
        self.ipr_names = _build_ipr_names()
        self.W = sig.build_distance_matrix(self.tree, TIPS)

    def _get_result(self, results, ipr_id):
        return next(r for r in results if r["ipr"] == ipr_id)

    def test_full_pipeline(self):
        results = sig.analyze(
            self.tree, TIPS, self.ipr_data, self.ipr_names,
            n_perm=N_PERM, do_continuous=True, W=self.W,
        )
        self.assertEqual(len(results), len(DOMAINS))

        # --- Invariant domains ---
        r = self._get_result(results, "IPR_INVARIANT_1")
        self.assertEqual(r["p_binary"], 1.0,
                         "Invariant (all 1) should have p_binary=1.0")

        r = self._get_result(results, "IPR_INVARIANT_0")
        self.assertEqual(r["p_binary"], 1.0,
                         "Invariant (all 0) should have p_binary=1.0")

        # --- Strong binary signal ---
        r = self._get_result(results, "IPR_CLADE_LEFT")
        self.assertEqual(r["parsimony"], 1)
        self.assertLess(r["p_binary"], 0.05,
                        f"IPR_CLADE_LEFT: expected significant binary, "
                        f"got p={r['p_binary']}")

        r = self._get_result(results, "IPR_CLADE_RIGHT")
        self.assertEqual(r["parsimony"], 1)
        self.assertLess(r["p_binary"], 0.05,
                        f"IPR_CLADE_RIGHT: expected significant binary, "
                        f"got p={r['p_binary']}")

        # --- No binary signal ---
        r = self._get_result(results, "IPR_SCATTERED")
        self.assertGreater(r["p_binary"], 0.1,
                           f"IPR_SCATTERED: should not be significant, "
                           f"got p={r['p_binary']}")

        # --- Singleton ---
        r = self._get_result(results, "IPR_SINGLETON")
        # With only 1 present out of 10, one change is expected by chance
        # so p should be high
        self.assertGreater(r["p_binary"], 0.3,
                           f"IPR_SINGLETON: single-tip should not be "
                           f"significant, got p={r['p_binary']}")

        # --- Strong continuous signal ---
        r = self._get_result(results, "IPR_COUNTS_CLADE")
        self.assertLess(r["p_continuous"], 0.05,
                        f"IPR_COUNTS_CLADE: expected significant continuous, "
                        f"got p={r['p_continuous']}")
        self.assertGreater(r["k_star"], 1.0,
                           f"IPR_COUNTS_CLADE: expected K*>1, "
                           f"got {r['k_star']}")

        # --- No continuous signal ---
        r = self._get_result(results, "IPR_COUNTS_NOISY")
        self.assertGreater(r["p_continuous"], 0.1,
                           f"IPR_COUNTS_NOISY: should not be significant, "
                           f"got p={r['p_continuous']}")


class TestPruningDistances(unittest.TestCase):
    """Verify that tree pruning preserves all pairwise distances.

    Uses a 16-tip tree where we keep only 5 tips spread across different
    clades.  This forces cascading unifurcation collapses — the scenario
    where Bio.Phylo.prune() loses branch lengths.
    """

    #      ┌─ a1:1         keep
    #   ┌──┤
    #   │  └─ a2:1         prune
    # ┌─┤:3
    # │ │  ┌─ a3:1         prune
    # │ └──┤
    # │    └─ a4:1         prune
    # ┤:5
    # │    ┌─ b1:2         prune
    # │ ┌──┤
    # │ │  └─ b2:2         keep
    # └─┤:3
    #   │  ┌─ b3:2         prune
    #   └──┤
    #      └─ b4:2         keep
    #
    # (plus a second major clade with c1..c4 and d1..d4)

    LARGE_NEWICK = (
        "("
          "(((a1:1,a2:1):3,(a3:1,a4:1):3):5,((b1:2,b2:2):3,(b3:2,b4:2):3):5):10,"
          "(((c1:1,c2:1):4,(c3:1,c4:1):4):6,((d1:2,d2:2):4,(d3:2,d4:2):4):6):10"
        ");"
    )
    ALL_TIPS = [f"{g}{i}" for g in "abcd" for i in "1234"]
    KEEP = {"a1", "b2", "b4", "c3", "d1"}   # one from each sub-clade

    def _write_tree(self, nwk):
        path = os.path.join(tempfile.gettempdir(), "_prune_test.nwk")
        with open(path, "w") as f:
            f.write(nwk)
        return path

    def test_pairwise_distances_preserved(self):
        """All pairwise distances between kept tips must match the
        original tree exactly."""
        path = self._write_tree(self.LARGE_NEWICK)
        t_orig = Phylo.read(path, "newick")

        # Record original distances
        orig_dist = {}
        for a in sorted(self.KEEP):
            for b in sorted(self.KEEP):
                if a < b:
                    orig_dist[(a, b)] = t_orig.distance(a, b)

        # Prune
        t_pruned = sig._prune_to_tips(t_orig, self.KEEP)

        # Verify
        for (a, b), d_orig in orig_dist.items():
            d_pruned = t_pruned.distance(a, b)
            self.assertAlmostEqual(
                d_orig, d_pruned, places=6,
                msg=f"{a}-{b}: orig={d_orig:.4f} pruned={d_pruned:.4f}"
            )

    def test_correct_tip_count(self):
        path = self._write_tree(self.LARGE_NEWICK)
        t_orig = Phylo.read(path, "newick")
        t_pruned = sig._prune_to_tips(t_orig, self.KEEP)
        tip_names = {t.name for t in t_pruned.get_terminals()}
        self.assertEqual(tip_names, self.KEEP)

    def test_single_tip_prune(self):
        """Pruning to one tip should work without error."""
        path = self._write_tree(self.LARGE_NEWICK)
        t = Phylo.read(path, "newick")
        t_pruned = sig._prune_to_tips(t, {"a1"})
        self.assertEqual(len(t_pruned.get_terminals()), 1)

    def test_no_matching_tips_raises(self):
        path = self._write_tree(self.LARGE_NEWICK)
        t = Phylo.read(path, "newick")
        with self.assertRaises(ValueError):
            sig._prune_to_tips(t, {"nonexistent"})

    def test_match_and_prune_preserves_distances(self):
        """Integration: match_and_prune with prefix also preserves
        distances."""
        # Add prefix
        prefixed_nwk = self.LARGE_NEWICK
        for tip in self.ALL_TIPS:
            prefixed_nwk = prefixed_nwk.replace(
                f"{tip}:", f"mycobiont_{tip}:")
        path = self._write_tree(prefixed_nwk)

        t_orig = Phylo.read(path, "newick")
        t_match, matched = sig.match_and_prune(
            Phylo.read(path, "newick"), self.KEEP, "mycobiont_"
        )
        self.assertEqual(matched, self.KEEP)

        # Check distances
        for a in sorted(self.KEEP):
            for b in sorted(self.KEEP):
                if a < b:
                    d_orig = t_orig.distance(f"mycobiont_{a}", f"mycobiont_{b}")
                    d_pruned = t_match.distance(a, b)
                    self.assertAlmostEqual(
                        d_orig, d_pruned, places=6,
                        msg=f"{a}-{b}: distance mismatch after match_and_prune"
                    )


class TestEndToEnd(unittest.TestCase):
    """End-to-end test: write files, run main-like pipeline, check outputs."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="ipr_test_")
        random.seed(42)
        np.random.seed(42)

        # Write test tree (with mycobiont_ prefix like real data)
        self.tree_path = os.path.join(self.tmpdir, "test.nwk")
        prefixed = TEST_NEWICK
        for tip in TIPS:
            prefixed = prefixed.replace(tip + ":", f"mycobiont_{tip}:")
            prefixed = prefixed.replace(tip + ")", f"mycobiont_{tip})")
        with open(self.tree_path, "w") as f:
            f.write(prefixed)

        # Write test matrix with annotation columns
        self.matrix_path = os.path.join(self.tmpdir, "test_matrix.tsv")
        ipr_data = _build_ipr_data()
        ipr_names = _build_ipr_names()
        with open(self.matrix_path, "w") as f:
            f.write("\t".join(
                ["IPR_ID"] + TIPS +
                ["ENTRY_AC", "ENTRY_TYPE", "ENTRY_NAME"]
            ) + "\n")
            for ipr_id in sorted(ipr_data.keys()):
                counts = [str(ipr_data[ipr_id][t]) for t in TIPS]
                name = ipr_names[ipr_id]
                f.write("\t".join(
                    [ipr_id] + counts +
                    [ipr_id, "Domain", name]
                ) + "\n")

        # Write mapping file (two "species")
        self.mapping_path = os.path.join(self.tmpdir, "mapping.tsv")
        with open(self.mapping_path, "w") as f:
            for tip in LEFT:
                f.write(f"{tip}\tSpecies_A\n")
            for tip in RIGHT:
                f.write(f"{tip}\tSpecies_B\n")
            f.write("s10\tOutgroup\n")

        self.outdir = os.path.join(self.tmpdir, "results")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_read_matrix_with_annotations(self):
        samples, ipr_data, ipr_names = sig.read_matrix(self.matrix_path)
        self.assertEqual(len(samples), 10)
        self.assertEqual(len(ipr_data), len(DOMAINS))
        # Check annotations were parsed
        self.assertEqual(ipr_names["IPR_CLADE_LEFT"], "Left-clade marker")
        self.assertNotIn("ENTRY_AC", samples,
                         "Annotation columns should not be in samples list")

    def test_tree_pruning(self):
        tree = Phylo.read(self.tree_path, "newick")
        samples, ipr_data, _ = sig.read_matrix(self.matrix_path)
        tree, matched = sig.match_and_prune(tree, set(samples), "mycobiont_")
        self.assertEqual(len(matched), 10)
        tip_names = {t.name for t in tree.get_terminals()}
        self.assertEqual(tip_names, set(TIPS))

    def test_full_pipeline_outputs(self):
        """Run the full pipeline and check all expected files are created."""
        tree = Phylo.read(self.tree_path, "newick")
        samples, ipr_data, ipr_names = sig.read_matrix(self.matrix_path)
        tree, matched = sig.match_and_prune(tree, set(samples), "mycobiont_")
        tip_order = [t.name for t in tree.get_terminals()]

        os.makedirs(self.outdir, exist_ok=True)

        W = sig.build_distance_matrix(tree, tip_order)
        results = sig.analyze(
            tree, tip_order, ipr_data, ipr_names,
            n_perm=99, do_continuous=True, W=W,
        )

        # Write TSV
        tsv_path = os.path.join(self.outdir, "signal.tsv")
        sig.write_results(results, tsv_path, include_continuous=True)
        self.assertTrue(os.path.exists(tsv_path))

        # Verify TSV has expected columns
        with open(tsv_path) as f:
            header = f.readline().strip().split("\t")
        self.assertIn("p_binary", header)
        self.assertIn("p_continuous", header)
        self.assertIn("k_star", header)
        self.assertIn("name", header)

        # Write SVGs
        for rank_by in ("binary", "continuous"):
            svg_path = os.path.join(self.outdir, f"heatmap_{rank_by}.svg")
            sig.render_svg(
                tree, tip_order, ipr_data, ipr_names, results,
                svg_path, top_n=9, rank_by=rank_by,
            )
            self.assertTrue(os.path.exists(svg_path))
            with open(svg_path) as f:
                content = f.read()
            self.assertIn("<svg", content)
            self.assertIn(f"ranked by", content)

    def test_unit_level_collapse(self):
        """Test sample-to-unit collapsing with mapping file."""
        tree = Phylo.read(self.tree_path, "newick")
        samples, ipr_data, _ = sig.read_matrix(self.matrix_path)
        tree, matched = sig.match_and_prune(tree, set(samples), "mycobiont_")
        mapping = sig.read_mapping(self.mapping_path)

        mapped = [s for s in TIPS if s in mapping]
        units, collapsed = sig.collapse_matrix(ipr_data, mapped, mapping)
        self.assertEqual(set(units), {"Species_A", "Species_B", "Outgroup"})

        # Check median collapsing for IPR_COUNTS_CLADE:
        # Species_A tips (s1-s5): counts = [8,9,7,10,8] → median = 8
        # Species_B tips (s6-s9): counts = [1,1,2,1]    → median = 1
        self.assertEqual(collapsed["IPR_COUNTS_CLADE"]["Species_A"], 8)
        self.assertEqual(collapsed["IPR_COUNTS_CLADE"]["Species_B"], 1)


# -----------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
