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

import numpy as np
import pytest
import SimpleITK as sitk

# MIST imports.
from mist.utils import sitk_io
from tests.regression.ants_sitk.fixtures import FIXTURES, generate_dataset

ants = pytest.importorskip("ants")


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
    """read_image + array_from_image vs. ants.image_read(...).numpy()."""

    def test_image_pixels_match(self, dataset, spec):
        path = _image_path(dataset, spec.patient_id)
        expected = ants.image_read(path).numpy()
        actual = sitk_io.array_from_image(sitk_io.read_image(path))
        np.testing.assert_array_equal(actual, expected)

    def test_mask_pixels_match(self, dataset, spec):
        path = _mask_path(dataset, spec.patient_id)
        expected = ants.image_read(path).numpy()
        actual = sitk_io.array_from_image(sitk_io.read_image(path))
        np.testing.assert_array_equal(actual, expected)


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
