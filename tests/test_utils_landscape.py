"""Tests for the free energy landscape maths and the analysis figure helpers."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

import utils


def two_wells(count=4000, seed=0):
    """Two Gaussian wells of unequal population, offset from each other."""
    generator = np.random.default_rng(seed)
    deep_x = generator.normal(-2.0, 0.3, count)
    deep_y = generator.normal(0.0, 0.3, count)
    shallow_x = generator.normal(2.0, 0.3, count // 4)
    shallow_y = generator.normal(1.0, 0.3, count // 4)
    return np.concatenate([deep_x, shallow_x]), np.concatenate([deep_y, shallow_y])


class FreeEnergyLandscapeTests(unittest.TestCase):
    def test_the_most_populated_bin_sits_at_zero(self):
        """G is a depth below the deepest well, so the minimum is exactly 0."""
        _, _, _, free_energy = utils.compute_free_energy_landscape(*two_wells(), bin_count=40)

        self.assertAlmostEqual(np.nanmin(free_energy), 0.0)
        self.assertGreater(np.nanmax(free_energy), 0.0)

    def test_unvisited_bins_are_nan_not_infinity(self):
        """+inf would collapse the colour scale onto one level; NaN renders blank."""
        _, _, probability, free_energy = utils.compute_free_energy_landscape(
            *two_wells(), bin_count=40)

        empty = probability == 0
        self.assertTrue(empty.any(), "the fixture should leave some bins empty")
        self.assertTrue(np.isnan(free_energy[empty]).all())
        self.assertFalse(np.isinf(free_energy[~empty]).any())

    def test_probability_is_normalised(self):
        _, _, probability, _ = utils.compute_free_energy_landscape(*two_wells(), bin_count=30)
        self.assertAlmostEqual(probability.sum(), 1.0)

    def test_temperature_scales_the_energy_difference(self):
        """G = -kT ln(P/Pmax), so doubling T doubles every depth."""
        x, y = two_wells()
        _, _, _, cold = utils.compute_free_energy_landscape(x, y, bin_count=30, temperature=150.0)
        _, _, _, hot = utils.compute_free_energy_landscape(x, y, bin_count=30, temperature=300.0)

        both = ~np.isnan(cold) & ~np.isnan(hot)
        np.testing.assert_allclose(hot[both], 2 * cold[both], rtol=1e-9)

    def test_a_known_ratio_gives_the_textbook_energy(self):
        """A bin visited a tenth as often as the deepest is kT ln(10) above it."""
        x = np.concatenate([np.full(1000, 0.25), np.full(100, 0.75)])
        y = np.concatenate([np.full(1000, 0.25), np.full(100, 0.75)])
        _, _, _, free_energy = utils.compute_free_energy_landscape(
            x, y, bin_count=2, temperature=300.0)

        expected = utils.BOLTZMANN_CONSTANT_KJ_PER_MOL_K * 300.0 * math.log(10)
        self.assertAlmostEqual(np.nanmax(free_energy), expected, places=6)

    def test_the_grid_is_indexed_x_then_y(self):
        """histogram2d returns [x, y]; the plot helper is what transposes it.

        Deliberately asymmetric in both axes: a symmetric fixture would pass
        whether or not the two indices were swapped.
        """
        # 100 points at (low x, low y), 10 at (high x, low y), 1 at (low x, high y).
        x = np.concatenate([np.full(100, 0.1), np.full(10, 0.9), np.full(1, 0.1)])
        y = np.concatenate([np.full(100, 0.1), np.full(10, 0.1), np.full(1, 0.9)])
        x_centres, y_centres, probability, _ = utils.compute_free_energy_landscape(
            x, y, bin_count=2)

        self.assertEqual(probability.shape, (2, 2))
        self.assertEqual(len(x_centres), 2)
        self.assertAlmostEqual(probability[0, 0], 100 / 111)
        self.assertAlmostEqual(probability[1, 0], 10 / 111)   # high x, low y
        self.assertAlmostEqual(probability[0, 1], 1 / 111)    # low x, high y
        self.assertAlmostEqual(probability[1, 1], 0.0)

    def test_an_empty_sample_is_rejected(self):
        with self.assertRaises(ValueError):
            utils.compute_free_energy_landscape([], [], bin_count=10)

    def test_invalid_configuration_and_nonfinite_coordinates_are_rejected(self):
        invalid_calls = (
            ([0.0], [0.0], 1, 300.0),
            ([0.0], [0.0], 10.5, 300.0),
            ([0.0], [0.0], 1001, 300.0),
            ([0.0], [0.0], 10, 0.0),
            ([0.0], [0.0], 10, float("inf")),
            ([0.0, float("nan")], [0.0, 1.0], 10, 300.0),
            ([0.0], [0.0, 1.0], 10, 300.0),
        )
        for x, y, bins, temperature in invalid_calls:
            with self.subTest(bins=bins, temperature=temperature, size=(len(x), len(y))):
                with self.assertRaises(ValueError):
                    utils.compute_free_energy_landscape(
                        x, y, bin_count=bins, temperature=temperature)


class LandscapeFigureTests(unittest.TestCase):
    def test_the_contour_grid_is_transposed_for_matplotlib(self):
        """contourf reads [row, column] as [y, x] but the grid is [x, y].

        Checked on a deliberately NON-SQUARE grid. On a square one a missing
        transpose silently mirrors the landscape about the diagonal and every
        assertion still passes; here it is a shape mismatch matplotlib rejects.
        """
        x_centres = np.linspace(-1.0, 1.0, 5)
        y_centres = np.linspace(0.0, 3.0, 7)
        free_energy = np.arange(35, dtype=float).reshape(5, 7)   # [x, y]

        figure = utils.make_landscape_figure(x_centres, y_centres, free_energy)
        self.assertGreater(len(figure.axes), 1)

    def test_the_minimum_marker_lands_on_the_deepest_well(self):
        x_centres, y_centres, _, free_energy = utils.compute_free_energy_landscape(
            *two_wells(), bin_count=25)
        figure = utils.make_landscape_figure(x_centres, y_centres, free_energy)

        # The deepest well of the fixture sits at x ~= -2, y ~= 0.
        marker = figure.axes[0].lines[0]
        self.assertLess(marker.get_xdata()[0], -1.0)
        self.assertAlmostEqual(marker.get_ydata()[0], 0.0, delta=0.5)

    def test_a_colourbar_is_attached(self):
        x_centres, y_centres, _, free_energy = utils.compute_free_energy_landscape(
            *two_wells(), bin_count=20)
        figure = utils.make_landscape_figure(x_centres, y_centres, free_energy,
                                             xlabel="PC1", ylabel="PC2", title="FEL")

        self.assertGreater(len(figure.axes), 1, "no colourbar axes were added")
        self.assertEqual(figure.axes[0].get_xlabel(), "PC1")
        self.assertEqual(figure.axes[0].get_title(), "FEL")

    def test_the_figure_never_touches_pyplot(self):
        import matplotlib.pyplot as plt
        before = plt.get_fignums()
        x_centres, y_centres, _, free_energy = utils.compute_free_energy_landscape(
            *two_wells(), bin_count=20)
        utils.make_landscape_figure(x_centres, y_centres, free_energy)
        self.assertEqual(plt.get_fignums(), before)


class PcaFigureTests(unittest.TestCase):
    def eigenvalues(self):
        return pd.DataFrame({"Eigenvector index": range(1, 11),
                             "(nm^2)": [10.0, 5.0, 2.5, 1.0, 0.6, 0.4, 0.3, 0.1, 0.05, 0.05]})

    def test_scree_plot_carries_bars_and_a_cumulative_curve(self):
        figure = utils.make_scree_figure(self.eigenvalues(), count=5, title="Scree")

        axes = figure.axes[0]
        self.assertEqual(len(axes.patches), 5, "expected one bar per eigenvalue shown")
        self.assertEqual(axes.get_title(), "Scree")
        share = figure.axes[1]
        # First eigenvalue is 10 of 20 total, so the curve starts at 50%.
        self.assertAlmostEqual(share.lines[0].get_ydata()[0], 50.0)
        self.assertEqual(share.get_ylim(), (0.0, 100.0))

    def test_scatter_is_coloured_by_row_order(self):
        frame = pd.DataFrame({"PC1": [0.0, 1.0, 2.0], "PC2": [1.0, 0.0, -1.0]})
        figure = utils.make_scatter_figure(frame, title="Projection")

        axes = figure.axes[0]
        self.assertEqual(axes.get_xlabel(), "PC1")
        self.assertEqual(axes.get_ylabel(), "PC2")
        self.assertEqual(len(axes.collections), 1)
        np.testing.assert_array_equal(axes.collections[0].get_array(), [0, 1, 2])

    def test_scree_rejects_zero_or_nonfinite_total_variance(self):
        for eigenvalues in ([0.0, 0.0], [1.0, float("nan")]):
            with self.subTest(eigenvalues=eigenvalues):
                frame = pd.DataFrame({"index": [1, 2], "value": eigenvalues})
                with self.assertRaises(ValueError):
                    utils.make_scree_figure(frame)


if __name__ == "__main__":
    unittest.main()
