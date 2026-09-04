"""Byte-for-byte regression tests: mist.utils.sitk_io vs. the ants equivalents.

Stage 1 release gate for the ANTs -> SimpleITK migration (see
ants_to_simpleitk_migration.md): every function in sitk_io must reproduce the
current ants-based output exactly on the Stage 0 fixtures, which stress the
axis-order/orientation edge cases the migration is actually worried about
(anisotropic spacing, oblique direction cosines, sparse labels).

Requires the `ants` package, which is not part of MIST's own dependency
declaration for this module (that's the point) and is heavy to install; the
whole file is skipped if it's not importable.
"""

import itertools

import numpy as np
import pytest
import SimpleITK as sitk

# MIST imports.
from mist.utils import sitk_io
from tests.regression.ants_sitk.fixtures import FIXTURES, generate_dataset

ants = pytest.importorskip("ants")


def _all_orientation_codes() -> list[str]:
    """All 48 valid three-letter anatomical orientation codes.

    Each code picks one letter from each of the three perpendicular axis
    pairs (Left/Right, Posterior/Anterior, Superior/Inferior) and assigns
    the three pairs to the three image axes in some order: 3! axis-order
    permutations x 2**3 letter choices = 48.
    """
    axis_pairs = [("L", "R"), ("P", "A"), ("S", "I")]
    return [
        "".join(pair[choice] for pair, choice in zip(perm, choices))
        for perm in itertools.permutations(axis_pairs)
        for choices in itertools.product(range(2), repeat=3)
    ]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Materialize the Stage 0 fixture patients once for this module."""
    root = tmp_path_factory.mktemp("sitk_io_fixtures")
    return generate_dataset(root)


def _image_path(dataset, patient_id: str) -> str:
    return str(dataset.train_data / patient_id / "image.nii.gz")


def _mask_path(dataset, patient_id: str) -> str:
    return str(dataset.train_data / patient_id / "mask.nii.gz")


@pytest.mark.parametrize("spec", FIXTURES, ids=lambda s: s.patient_id)
class TestReadImage:
    """read_image + array_from_image vs. ants.image_read(...).numpy().

    Also asserts dtype, not just values: ants.image_read defaults to
    pixeltype="float", force-casting every image on read regardless of its
    on-disk type (see read_image's docstring) -- assert_array_equal alone
    would pass even if that cast were missing, since e.g. uint8(2) ==
    float32(2.0). This is exactly the class of thing that class of
    assertion silently hides, so dtype is checked explicitly.
    """

    def test_image_pixels_match(self, dataset, spec):
        path = _image_path(dataset, spec.patient_id)
        expected = ants.image_read(path).numpy()
        actual = sitk_io.array_from_image(sitk_io.read_image(path))
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype

    def test_mask_pixels_match(self, dataset, spec):
        path = _mask_path(dataset, spec.patient_id)
        expected = ants.image_read(path).numpy()
        actual = sitk_io.array_from_image(sitk_io.read_image(path))
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype


@pytest.mark.parametrize("spec", FIXTURES, ids=lambda s: s.patient_id)
class TestReadImageHeader:
    """read_image_header vs. ants.image_header_info."""

    def test_header_matches(self, dataset, spec):
        path = _image_path(dataset, spec.patient_id)
        expected = ants.image_header_info(path)
        actual = sitk_io.read_image_header(path)

        assert actual["dimensions"] == expected["dimensions"]
        assert actual["origin"] == expected["origin"]
        # Spacing and direction are compared with a tolerance, matching
        # analyzer_utils.compare_headers' own policy: ants and SimpleITK
        # parse a rotated (non-axis-aligned) NIfTI sform into spacing via
        # different floating-point paths (pixdim directly vs. sform column
        # norms), so exact equality isn't guaranteed even though both are
        # reading the same on-disk float32 header fields.
        np.testing.assert_allclose(actual["spacing"], expected["spacing"], atol=1e-6)
        # ants.image_header_info's direction is visibly lower-precision than
        # a full image read's (see sitk_io.read_image_header's docstring) --
        # atol here is looser than the 1e-10 used elsewhere in this file for
        # exactly that reason, not because sitk_io is imprecise.
        np.testing.assert_allclose(actual["direction"], expected["direction"], atol=1e-4)


@pytest.mark.parametrize("spec", FIXTURES, ids=lambda s: s.patient_id)
class TestImageFromArray:
    """image_from_array vs. ants.from_numpy, round-tripped through disk."""

    def test_round_trip_matches(self, dataset, spec, tmp_path):
        path = _image_path(dataset, spec.patient_id)
        array = ants.image_read(path).numpy()
        spacing = spec.spacing_xyz
        origin = spec.origin_xyz
        direction = np.reshape(np.array(spec.direction), (3, 3))

        ants_img = ants.from_numpy(array, spacing=spacing, origin=origin, direction=direction)
        ants_path = tmp_path / "from_ants.nii.gz"
        ants.image_write(ants_img, str(ants_path))

        sitk_img = sitk_io.image_from_array(
            array, spacing=spacing, origin=origin, direction=direction
        )
        sitk_path = tmp_path / "from_sitk.nii.gz"
        sitk_io.write_image(sitk_img, str(sitk_path))

        expected = ants.image_read(str(ants_path))
        actual = ants.image_read(str(sitk_path))
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        assert actual.spacing == expected.spacing
        assert actual.origin == expected.origin
        np.testing.assert_allclose(actual.direction, expected.direction, atol=1e-10)


@pytest.mark.parametrize("spec", FIXTURES, ids=lambda s: s.patient_id)
class TestReorientImage:
    """reorient_image vs. ants.reorient_image2, including the oblique case."""

    @pytest.mark.parametrize("orientation", ["RAI", "LPI", "ASL"])
    def test_reoriented_pixels_match(self, dataset, spec, orientation):
        path = _image_path(dataset, spec.patient_id)

        ants_img = ants.image_read(path)
        expected = ants.reorient_image2(ants_img, orientation)

        sitk_img = sitk_io.read_image(path)
        actual = sitk_io.reorient_image(sitk_img, orientation)

        np.testing.assert_array_equal(sitk_io.array_from_image(actual), expected.numpy())
        assert actual.GetSize() == expected.shape
        np.testing.assert_allclose(actual.GetSpacing(), expected.spacing, atol=1e-10)


@pytest.mark.parametrize("spec", FIXTURES, ids=lambda s: s.patient_id)
class TestGetOrientation:
    """get_orientation vs. ants.get_orientation."""

    def test_orientation_string_matches(self, dataset, spec):
        path = _image_path(dataset, spec.patient_id)
        expected = ants.get_orientation(ants.image_read(path))
        actual = sitk_io.get_orientation(sitk_io.read_image(path))
        assert actual == expected


@pytest.mark.parametrize("spec", FIXTURES, ids=lambda s: s.patient_id)
class TestCropImage:
    """crop_image vs. ants.crop_indices (index/size, not lowerind/upperind)."""

    def test_crop_matches(self, dataset, spec):
        path = _image_path(dataset, spec.patient_id)
        size_x, size_y, size_z = spec.size_xyz
        lowerind = [size_x // 6, size_y // 6, size_z // 6]
        upperind = [size_x - size_x // 6, size_y - size_y // 6, size_z - size_z // 6]
        crop_size = [upper - lower for upper, lower in zip(upperind, lowerind)]

        expected = ants.crop_indices(ants.image_read(path), lowerind=lowerind, upperind=upperind)
        actual = sitk_io.crop_image(sitk_io.read_image(path), index=lowerind, size=crop_size)

        np.testing.assert_array_equal(sitk_io.array_from_image(actual), expected.numpy())


@pytest.mark.parametrize("spec", FIXTURES, ids=lambda s: s.patient_id)
class TestPadImage:
    """pad_image vs. ants.pad_image."""

    def test_pad_matches(self, dataset, spec):
        path = _image_path(dataset, spec.patient_id)
        lower_padding = [2, 0, 3]
        upper_padding = [1, 4, 0]

        expected = ants.pad_image(
            ants.image_read(path),
            pad_width=list(zip(lower_padding, upper_padding)),
            return_padvals=False,
        )
        actual = sitk_io.pad_image(
            sitk_io.read_image(path), lower_padding=lower_padding, upper_padding=upper_padding
        )

        np.testing.assert_array_equal(sitk_io.array_from_image(actual), expected.numpy())


@pytest.mark.parametrize("spec", FIXTURES, ids=lambda s: s.patient_id)
class TestMergeChannels:
    """merge_channels vs. ants.merge_channels."""

    def test_merge_matches(self, dataset, spec):
        path = _image_path(dataset, spec.patient_id)

        ants_img = ants.image_read(path)
        expected = ants.merge_channels([ants_img, ants_img * 2.0]).numpy()

        sitk_img = sitk_io.read_image(path)
        doubled = sitk_img * 2.0
        merged = sitk_io.merge_channels([sitk_img, doubled])
        # merge_channels' output is multi-component, which array_from_image
        # explicitly doesn't support (see its docstring) -- convert directly
        # here, keeping the component axis last to match ants' (x, y, z, c).
        actual = sitk.GetArrayFromImage(merged).transpose(2, 1, 0, 3)

        np.testing.assert_array_equal(actual, expected)


class TestNonexistentPath:
    """read_image / read_image_header on a missing file: both must fail
    loudly, matching ants' behavior, rather than one of them returning
    something garbage. Exception *type* legitimately differs between the two
    libraries (ants raises ValueError/Exception, SimpleITK raises
    RuntimeError), so only "raises" is asserted, not a specific type.
    """

    def test_read_image_raises(self):
        with pytest.raises(Exception):  # noqa: B017, PT011
            ants.image_read("/nonexistent/path/does_not_exist.nii.gz")
        with pytest.raises(Exception):  # noqa: B017, PT011
            sitk_io.read_image("/nonexistent/path/does_not_exist.nii.gz")

    def test_read_image_header_raises(self):
        with pytest.raises(Exception):  # noqa: B017, PT011
            ants.image_header_info("/nonexistent/path/does_not_exist.nii.gz")
        with pytest.raises(Exception):  # noqa: B017, PT011
            sitk_io.read_image_header("/nonexistent/path/does_not_exist.nii.gz")


class TestCropImageEdgeCases:
    """crop_image edge cases: out-of-bounds/negative index, and the
    zero-size divergence from ants.crop_indices that crop_image now guards
    against explicitly (see its docstring).
    """

    def test_out_of_bounds_raises(self, dataset):
        path = _image_path(dataset, "iso_small")
        with pytest.raises(Exception):  # noqa: B017, PT011
            ants.crop_indices(
                ants.image_read(path), lowerind=[0, 0, 0], upperind=[1000, 1000, 1000]
            )
        with pytest.raises(Exception):  # noqa: B017, PT011
            sitk_io.crop_image(sitk_io.read_image(path), index=[0, 0, 0], size=[1000, 1000, 1000])

    def test_negative_index_raises(self, dataset):
        path = _image_path(dataset, "iso_small")
        with pytest.raises(Exception):  # noqa: B017, PT011
            ants.crop_indices(ants.image_read(path), lowerind=[-1, 0, 0], upperind=[2, 2, 2])
        with pytest.raises(Exception):  # noqa: B017, PT011
            sitk_io.crop_image(sitk_io.read_image(path), index=[-1, 0, 0], size=[3, 2, 2])

    def test_zero_size_raises_like_ants(self, dataset):
        """ants.crop_indices raises on a zero-sized crop; a bare
        sitk.RegionOfInterest call does not (it silently returns a
        degenerate empty-dimension image). crop_image adds its own guard to
        close that gap -- this test locks in that it actually raises.
        """
        path = _image_path(dataset, "iso_small")
        with pytest.raises(Exception):  # noqa: B017, PT011
            ants.crop_indices(ants.image_read(path), lowerind=[0, 0, 0], upperind=[0, 2, 2])
        with pytest.raises(ValueError):
            sitk_io.crop_image(sitk_io.read_image(path), index=[0, 0, 0], size=[0, 2, 2])


class TestPadImageEdgeCases:
    """pad_image edge cases: zero padding (no-op), and the negative-padding
    divergence from ants.pad_image that pad_image now guards against
    explicitly (see its docstring).
    """

    def test_zero_padding_is_noop(self, dataset):
        path = _image_path(dataset, "iso_small")
        expected = ants.pad_image(
            ants.image_read(path), pad_width=[(0, 0), (0, 0), (0, 0)], return_padvals=False
        )
        actual = sitk_io.pad_image(
            sitk_io.read_image(path), lower_padding=[0, 0, 0], upper_padding=[0, 0, 0]
        )
        np.testing.assert_array_equal(sitk_io.array_from_image(actual), expected.numpy())

    def test_negative_padding_raises_instead_of_silently_cropping(self, dataset):
        """ants.pad_image silently reinterprets a negative value as a crop
        (see pad_image's docstring); pad_image raises instead.
        """
        path = _image_path(dataset, "iso_small")
        with pytest.raises(ValueError):
            sitk_io.pad_image(
                sitk_io.read_image(path), lower_padding=[-1, 0, 0], upper_padding=[0, 0, 0]
            )


class TestImageFromArrayDefaults:
    """image_from_array with no spacing/origin/direction matches
    ants.from_numpy's own defaults (identity direction, zero origin, unit
    spacing)."""

    def test_defaults_match_ants(self):
        array = np.zeros((2, 3, 4), dtype=np.float32)
        expected = ants.from_numpy(array)
        actual = sitk_io.image_from_array(array)

        assert actual.GetSpacing() == expected.spacing
        assert actual.GetOrigin() == expected.origin
        np.testing.assert_array_equal(
            np.reshape(np.array(actual.GetDirection()), (3, 3)), expected.direction
        )


class TestMergeChannelsEdgeCases:
    """merge_channels edge cases: a single image (degenerate but valid),
    and the mismatched-geometry divergence from ants.merge_channels that
    merge_channels deliberately does not replicate (see its docstring).
    """

    def test_single_image(self, dataset):
        """A single-image merge is where ants and SimpleITK's array shapes
        diverge in a way that's specific to the single-image case: ants
        keeps an explicit trailing size-1 channel axis on `.numpy()` even
        for a 1-component merge, while sitk.Compose reports
        GetNumberOfComponentsPerPixel() == 1 and array_from_image's scalar
        path (correctly) returns no trailing axis at all. MIST's own call
        sites never merge a single channel (always the full class count, at
        least 2), so this is exercised here for completeness rather than
        because anything in mist/ depends on it.
        """
        path = _image_path(dataset, "iso_small")
        expected = ants.merge_channels([ants.image_read(path)]).numpy()
        assert expected.shape[-1] == 1

        merged = sitk_io.merge_channels([sitk_io.read_image(path)])
        assert merged.GetNumberOfComponentsPerPixel() == 1
        actual = sitk_io.array_from_image(merged)

        np.testing.assert_array_equal(actual, expected[..., 0])

    def test_mismatched_geometry_raises_instead_of_silently_succeeding(self, dataset):
        """ants.merge_channels on mismatched geometry silently succeeds
        using the first image's geometry (see merge_channels' docstring);
        merge_channels raises instead.
        """
        path = _image_path(dataset, "iso_small")
        full = sitk_io.read_image(path)
        mismatched = sitk_io.crop_image(full, index=[0, 0, 0], size=[2, 2, 2])
        with pytest.raises(RuntimeError):
            sitk_io.merge_channels([full, mismatched])


class TestOrientationCodeSweep:
    """Exhaustive sweep of reorient_image/get_orientation over all 48 valid
    orientation codes against ants, on one representative fixture.

    Fixture geometry diversity (anisotropic, oblique, sparse labels) is
    already covered by TestReorientImage's 3-code x 4-fixture matrix; this
    class is specifically about whether _ANTS_SITK_ORIENTATION_LETTER_FLIP
    generalizes across the full set of codes, which is a property of the
    codes themselves, not of image content -- so one fixture is enough.
    """

    @pytest.mark.parametrize("orientation", _all_orientation_codes())
    def test_reorient_matches_ants(self, dataset, orientation):
        path = _image_path(dataset, "iso_small")
        expected = ants.reorient_image2(ants.image_read(path), orientation)
        actual = sitk_io.reorient_image(sitk_io.read_image(path), orientation)
        np.testing.assert_array_equal(sitk_io.array_from_image(actual), expected.numpy())

    @pytest.mark.parametrize("orientation", _all_orientation_codes())
    def test_get_orientation_round_trips(self, dataset, orientation):
        path = _image_path(dataset, "iso_small")
        reoriented = sitk_io.reorient_image(sitk_io.read_image(path), orientation)
        assert sitk_io.get_orientation(reoriented) == orientation


class TestImageFromArrayNonContiguous:
    """image_from_array on a non-contiguous array (a Fortran-ordered array,
    and a strided slice/view) matches both a contiguous array with the same
    values and ants.from_numpy on the same non-contiguous array. Real call
    sites can hand this a view into a larger array (e.g. a per-class slice
    of a probability volume), not necessarily its own contiguous buffer.
    """

    def test_fortran_ordered_array_matches(self, dataset):
        path = _image_path(dataset, "anisotropic")
        contiguous = ants.image_read(path).numpy()
        fortran = np.asfortranarray(contiguous)
        assert not fortran.flags["C_CONTIGUOUS"]

        expected = ants.from_numpy(fortran, spacing=(1.0, 1.0, 1.0)).numpy()
        actual = sitk_io.array_from_image(
            sitk_io.image_from_array(fortran, spacing=(1.0, 1.0, 1.0))
        )
        np.testing.assert_array_equal(actual, expected)

    def test_strided_view_matches(self, dataset):
        path = _image_path(dataset, "anisotropic")
        contiguous = ants.image_read(path).numpy()
        # A slice-with-step view: same values, non-contiguous, non-Fortran
        # strides -- a different non-contiguity shape than the Fortran case.
        padded = np.zeros(tuple(2 * s for s in contiguous.shape), dtype=contiguous.dtype)
        padded[::2, ::2, ::2] = contiguous
        view = padded[::2, ::2, ::2]
        assert not view.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(view, contiguous)

        expected = ants.from_numpy(view, spacing=(1.0, 1.0, 1.0)).numpy()
        actual = sitk_io.array_from_image(sitk_io.image_from_array(view, spacing=(1.0, 1.0, 1.0)))
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("spec", FIXTURES, ids=lambda s: s.patient_id)
class TestCropPadRoundTrip:
    """Identity crop (whole image) and pad-then-crop-back round trips.

    These stress the exclusive-upperind-derived size arithmetic in
    crop_image directly, independent of comparing against ants -- an
    off-by-one here wouldn't necessarily show up as a mismatch against ants
    if both happened to be off the same way, but it would show up as a
    failure to round-trip back to the original data.
    """

    def test_identity_crop_returns_same_data(self, dataset, spec):
        path = _image_path(dataset, spec.patient_id)
        image = sitk_io.read_image(path)
        cropped = sitk_io.crop_image(image, index=[0, 0, 0], size=list(spec.size_xyz))
        np.testing.assert_array_equal(
            sitk_io.array_from_image(cropped), sitk_io.array_from_image(image)
        )

    def test_pad_then_crop_back_returns_original_data(self, dataset, spec):
        path = _image_path(dataset, spec.patient_id)
        image = sitk_io.read_image(path)
        original = sitk_io.array_from_image(image)

        lower_padding = [2, 0, 3]
        upper_padding = [1, 4, 0]
        padded = sitk_io.pad_image(image, lower_padding=lower_padding, upper_padding=upper_padding)
        cropped_back = sitk_io.crop_image(padded, index=lower_padding, size=list(spec.size_xyz))

        np.testing.assert_array_equal(sitk_io.array_from_image(cropped_back), original)


class TestImageFromArrayDtype:
    """image_from_array on an integer (label-like) array preserves dtype,
    matching ants.from_numpy -- unlike read_image (see its docstring),
    from_numpy has no pixeltype-forcing default to replicate; it just
    preserves whatever dtype the input array already has.
    """

    def test_uint8_array_preserves_dtype(self, dataset):
        path = _mask_path(dataset, "sparse_labels")
        array = sitk_io.array_from_image(sitk_io.read_image(path, pixeltype="unsigned char"))
        assert array.dtype == np.uint8

        expected = ants.from_numpy(array, spacing=(1.0, 1.0, 1.0))
        actual = sitk_io.image_from_array(array, spacing=(1.0, 1.0, 1.0))

        assert expected.numpy().dtype == np.uint8
        assert sitk_io.array_from_image(actual).dtype == np.uint8
        np.testing.assert_array_equal(sitk_io.array_from_image(actual), expected.numpy())


class TestReadImagePixeltypes:
    """read_image's pixeltype table has four entries; TestReadImage only
    exercises the default ("float") and TestImageFromArrayDtype separately
    exercises "unsigned char" -- this covers the other two ("unsigned int",
    "double") against ants.image_read with the same pixeltype, so the whole
    table is verified rather than half of it being trusted on the pattern
    of the entries that were checked.
    """

    @pytest.mark.parametrize(
        "pixeltype,expected_dtype",
        [("unsigned int", np.uint32), ("double", np.float64)],
    )
    def test_pixeltype_matches_ants(self, dataset, pixeltype, expected_dtype):
        path = _image_path(dataset, "anisotropic")
        expected = ants.image_read(path, pixeltype=pixeltype).numpy()
        actual = sitk_io.array_from_image(sitk_io.read_image(path, pixeltype=pixeltype))

        assert expected.dtype == expected_dtype
        assert actual.dtype == expected_dtype
        np.testing.assert_array_equal(actual, expected)
