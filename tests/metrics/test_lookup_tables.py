"""Unit tests for the lookup_tables module used in surface distance metrics."""
import numpy as np

# MIST imports.
from mist.metrics.lookup_tables import (
    create_table_neighbour_code_to_surface_area,
    create_table_neighbour_code_to_contour_length,
)


def test_create_table_neighbour_code_to_surface_area():
    """Should return a (256,) array of non-negative floats."""
    spacing_mm = (1.0, 1.0, 1.0)
    table = create_table_neighbour_code_to_surface_area(spacing_mm)

    assert isinstance(table, np.ndarray)
    assert table.shape == (256,)
    assert table.dtype == np.float64
    assert np.all(table >= 0.0)


def test_create_table_neighbour_code_to_contour_length_shape_and_dtype():
    """Should return a (16,) array of non-negative floats."""
    spacing_mm = (1.0, 1.0)
    table = create_table_neighbour_code_to_contour_length(spacing_mm)

    assert isinstance(table, np.ndarray)
    assert table.shape == (16,)
    assert table.dtype == np.float64
    assert np.all(table >= 0.0)


def test_create_table_neighbour_code_to_contour_length_values():
    """Should correctly compute known contour lengths for 2D cases."""
    spacing_mm = (1.0, 1.0)
    table = create_table_neighbour_code_to_contour_length(spacing_mm)

    diag = 0.5 * np.sqrt(2)
    vertical = 1.0
    horizontal = 1.0

    assert np.isclose(table[int("0001", 2)], diag)  # Bottom right only.
    assert np.isclose(table[int("0011", 2)], horizontal)  # Top row.
    assert np.isclose(table[int("0101", 2)], vertical)  # Left column.
    assert np.isclose(table[int("0110", 2)], 2 * diag)  # Cross pattern.
    assert np.isclose(table[int("1111", 2)], 0.0)  # Fully filled -> no boundary.
