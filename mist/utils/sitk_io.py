"""SimpleITK-based replacements for the ANTs functions MIST currently uses.

Stage 1 of the ANTs -> SimpleITK migration (see ants_to_simpleitk_migration.md).
This module is not wired into the pipeline yet; later stages replace each
`ants.*` call site with the corresponding function here. Keeping the
(x, y, z) <-> (z, y, x) array-axis conversion in exactly two places
(`image_from_array`/`array_from_image`) is the point: everywhere else in
MIST's business logic (bounding boxes, target_spacing tuples, crop/pad
indices) is written in ANTs' (x, y, z) convention, and a transpose bug
anywhere else would be silent, not a crash.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk

# ants.image_read's signature is `image_read(filename, dimension=None,
# pixeltype='float', reorient=False)` -- it force-casts every image to this
# pixel type on read regardless of the file's on-disk type, and every
# ants.image_read(...) call site in mist/ uses that "float" default (none
# pass pixeltype explicitly). Verified empirically: a mask file whose NIfTI
# header genuinely says uint8 (confirmed via plain sitk.ReadImage, which
# preserves it) still comes back as float32 from ants.image_read. A plain
# sitk.ReadImage does NOT do this -- it preserves the on-disk pixel type --
# so matching ants' actual behavior requires an explicit cast here, not just
# a passthrough read.
_PIXELTYPE_TO_SITK = {
    "unsigned char": sitk.sitkUInt8,
    "unsigned int": sitk.sitkUInt32,
    "float": sitk.sitkFloat32,
    "double": sitk.sitkFloat64,
}


def read_image(path: str | Path, pixeltype: str = "float") -> sitk.Image:
    """Read an image from disk.

    Replacement for `ants.image_read`. Defaults to casting to float32,
    matching ants.image_read's own default -- see the module-level note on
    `_PIXELTYPE_TO_SITK` for why this isn't just `sitk.ReadImage`.

    Args:
        path: Path to the image file.
        pixeltype: Pixel type to cast to: "unsigned char", "unsigned int",
            "float", or "double", matching ants.image_read's own options.

    Returns:
        The image, cast to `pixeltype`.
    """
    image = sitk.ReadImage(str(path))
    return sitk.Cast(image, _PIXELTYPE_TO_SITK[pixeltype])


def write_image(image: sitk.Image, path: str | Path) -> None:
    """Write an image to disk.

    Replacement for `ants.image_write`.

    Args:
        image: Image to write.
        path: Destination path.
    """
    sitk.WriteImage(image, str(path))


def read_image_header(path: str | Path) -> dict[str, Any]:
    """Read an image's header without loading its pixel data.

    Replacement for `ants.image_header_info`. Uses a lazy, header-only read
    so callers that only need geometry (e.g., dataset analysis) don't pay for
    a full pixel-data load.

    Args:
        path: Path to the image file.

    Returns:
        Dictionary with the subset of ants.image_header_info's keys that MIST
        actually reads:
            - "dimensions": image size in (x, y, z) index order, as a tuple
              of floats (matching ants.image_header_info's own type).
            - "spacing": voxel spacing, (x, y, z) order.
            - "origin": image origin, (x, y, z) order.
            - "direction": (ndim, ndim) direction cosine matrix, in ants'
              image_header_info convention (see note below).
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()

    # Verified empirically (ants 0.6.1): ants.image_header_info's "direction"
    # is the *transpose* of the direction on the ANTsImage object you get
    # from ants.image_read(...).direction -- image_header_info appears to
    # take a lower-precision path internally (plausibly qform-derived vs.
    # the full read's sform-derived direction), which SimpleITK's
    # GetDirection() matches for row-vector convention. Transposing here
    # reproduces ants.image_header_info's specific convention; do not
    # "simplify" this to match image_from_array/array_from_image's
    # convention, which is deliberately a *different* ants direction
    # convention (the ANTsImage attribute's, not image_header_info's).
    dim = reader.GetDimension()
    direction = np.reshape(np.array(reader.GetDirection()), (dim, dim)).T
    return {
        "dimensions": tuple(float(s) for s in reader.GetSize()),
        "spacing": reader.GetSpacing(),
        "origin": reader.GetOrigin(),
        "direction": direction,
    }


def image_from_array(
    array: npt.NDArray[Any],
    spacing: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    direction: npt.NDArray[Any] | Sequence[float] | None = None,
) -> sitk.Image:
    """Build a scalar image from an (x, y, z)-ordered array.

    Replacement for `ants.from_numpy`. `array` is expected in the same
    (x, y, z) axis order ANTs uses; SimpleITK's own array order is (z, y, x),
    so this applies `array.T` once, here, rather than leaving that transpose
    to be re-derived ad hoc at each call site.

    Args:
        array: Image data in (x, y, z) axis order.
        spacing: Optional voxel spacing, (x, y, z) order. Defaults to
            SimpleITK's own default (1.0 per axis) if not given.
        origin: Optional image origin, (x, y, z) order. Defaults to
            SimpleITK's own default (0.0 per axis) if not given.
        direction: Optional (ndim, ndim) direction cosine matrix, or a flat
            row-major sequence of ndim * ndim values. Defaults to SimpleITK's
            own default (identity) if not given.

    Returns:
        The resulting image.
    """
    image = sitk.GetImageFromArray(array.T)
    if spacing is not None:
        image.SetSpacing(tuple(float(s) for s in spacing))
    if origin is not None:
        image.SetOrigin(tuple(float(o) for o in origin))
    if direction is not None:
        image.SetDirection(tuple(float(d) for d in np.asarray(direction).flatten()))
    return image


def array_from_image(image: sitk.Image) -> npt.NDArray[Any]:
    """Get an (x, y, z)-ordered array from a scalar image.

    Replacement for `img.numpy()`. SimpleITK's native array order is
    (z, y, x); this applies `.T` once to match the (x, y, z) order ANTs uses.

    Only valid for single-component (scalar) images: for a multi-component
    image (e.g., the output of `merge_channels`), a blanket `.T` would also
    reverse the component axis instead of keeping it last, which does not
    match ants' (x, y, z, c) convention. MIST doesn't currently call this on
    multi-component images; if a future stage needs to, it needs its own
    conversion, not this function.

    Args:
        image: Image to convert. Must have a single component per voxel.

    Returns:
        The image data in (x, y, z) axis order.
    """
    return sitk.GetArrayFromImage(image).T


# ants' orientation codes (e.g. "RAI") and SimpleITK's native
# DICOMOrientImageFilter codes (e.g. "LPS" for that same direction) describe
# the same physical direction with an opposite per-axis letter: ants follows
# the RAS-based (NIfTI/FSL-world) convention, SimpleITK's DICOMOrientImageFilter
# follows the LPS-based (DICOM/ITK-world) convention. Verified empirically
# (against ants 0.6.1 / SimpleITK 2.5.2) that the codes are related by a
# constant per-letter mirror, not by anything position- or image-dependent:
# an identity direction reads "RAI" in ants and "LPS" in SimpleITK, and that
# mapping holds for arbitrary axis-aligned and oblique directions alike. This
# table translates between the two so this module's public orientation API
# matches ants' convention, which is what the "RAI" literals hardcoded
# elsewhere in MIST assume.
_ANTS_SITK_ORIENTATION_LETTER_FLIP = {
    "R": "L",
    "L": "R",
    "A": "P",
    "P": "A",
    "I": "S",
    "S": "I",
}


def _flip_orientation_convention(code: str) -> str:
    """Translate an orientation code between the ants and SimpleITK conventions.

    The mapping is its own inverse, so this is used in both directions.
    """
    return "".join(_ANTS_SITK_ORIENTATION_LETTER_FLIP[c] for c in code)


def reorient_image(image: sitk.Image, orientation: str = "RAI") -> sitk.Image:
    """Reorient an image to the given anatomical orientation code.

    Replacement for `ants.reorient_image2`. `orientation` is in ants'
    convention (e.g. "RAI"), matching the literals used elsewhere in MIST;
    see the module-level note on `_ANTS_SITK_ORIENTATION_LETTER_FLIP` for why
    this isn't just `sitk.DICOMOrient`.

    Args:
        image: Image to reorient.
        orientation: Three-letter anatomical orientation code, in ants'
            convention (e.g. "RAI").

    Returns:
        The reoriented image.
    """
    return sitk.DICOMOrient(image, _flip_orientation_convention(orientation))


def get_orientation(image: sitk.Image) -> str:
    """Get an image's anatomical orientation code from its direction cosines.

    Replacement for `ants.get_orientation`. Returns the code in ants'
    convention (e.g. "RAI"); see the module-level note on
    `_ANTS_SITK_ORIENTATION_LETTER_FLIP` for why this isn't just
    `sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines`.

    Args:
        image: Image to inspect.

    Returns:
        Three-letter anatomical orientation code, in ants' convention (e.g.
        "RAI").
    """
    sitk_code = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection())
    return _flip_orientation_convention(sitk_code)


def crop_image(
    image: sitk.Image,
    index: Sequence[int],
    size: Sequence[int],
) -> sitk.Image:
    """Extract a sub-image region.

    Replacement for `ants.crop_indices`. Takes SimpleITK's native
    (index, size) parameterization rather than ants.crop_indices'
    (lowerind, upperind) — verified empirically that ants.crop_indices'
    upperind is exclusive, i.e. size = upperind - lowerind, so a caller
    migrating a `crop_indices(lowerind, upperind)` call should pass
    `index=lowerind, size=upperind - lowerind` here.

    A zero (or negative) entry in `size` raises, matching ants.crop_indices'
    behavior: verified empirically that ants.crop_indices raises on a
    zero-sized crop, while sitk.RegionOfInterest on its own silently returns
    a degenerate empty-dimension image. That divergence is exactly the class
    of silent failure this migration exists to avoid, so it's guarded
    against explicitly here rather than left to whatever RegionOfInterest
    happens to do.

    Args:
        image: Image to crop.
        index: Starting voxel index, (x, y, z) order.
        size: Extracted region size, (x, y, z) order. Every entry must be
            positive.

    Returns:
        The cropped image.

    Raises:
        ValueError: If any entry of `size` is not positive.
    """
    size = [int(s) for s in size]
    if any(s <= 0 for s in size):
        raise ValueError(f"crop_image requires a positive size on every axis, got {size}.")
    return sitk.RegionOfInterest(
        image,
        size=size,
        index=[int(i) for i in index],
    )


def pad_image(
    image: sitk.Image,
    lower_padding: Sequence[int],
    upper_padding: Sequence[int],
    constant: float = 0.0,
) -> sitk.Image:
    """Pad an image with a constant value.

    Replacement for `ants.pad_image`. Takes explicit (lower, upper) padding
    per axis instead of ants.pad_image's list-of-tuples `pad_width`, since
    ants.pad_image also supports a "pad to a given shape" mode that MIST
    doesn't use; MIST only ever pads by explicit per-axis amounts.

    Negative padding raises. Verified empirically that ants.pad_image
    accepts a negative value and silently reinterprets it as a *crop*
    instead of a pad (e.g. lower_padding=-1 trims a voxel off that axis).
    That's a surprising enough semantic that MIST's own call site already
    clamps to non-negative before calling it (see
    inference_utils.decrop_from_fg), so this deliberately does not replicate
    it — better to fail loudly on a negative value than silently start
    cropping where a pad was intended.

    Args:
        image: Image to pad.
        lower_padding: Voxels to add before the image, (x, y, z) order. Every
            entry must be non-negative.
        upper_padding: Voxels to add after the image, (x, y, z) order. Every
            entry must be non-negative.
        constant: Fill value for padded voxels.

    Returns:
        The padded image.

    Raises:
        ValueError: If any padding entry is negative.
    """
    lower_padding = [int(p) for p in lower_padding]
    upper_padding = [int(p) for p in upper_padding]
    if any(p < 0 for p in lower_padding) or any(p < 0 for p in upper_padding):
        raise ValueError(
            "pad_image requires non-negative padding on every axis, got "
            f"lower_padding={lower_padding}, upper_padding={upper_padding}."
        )
    return sitk.ConstantPad(image, lower_padding, upper_padding, constant)


def merge_channels(images: Sequence[sitk.Image]) -> sitk.Image:
    """Merge scalar images into one multi-component image.

    Replacement for `ants.merge_channels`. Component order matches input
    order.

    Deliberately does not replicate one piece of ants.merge_channels'
    behavior: verified empirically that ants.merge_channels on images with
    mismatched geometry silently succeeds, using the first image's geometry
    and apparently ignoring the mismatch, whereas sitk.Compose raises. A
    geometry mismatch here almost certainly means a bug upstream (e.g. two
    models' outputs that were never actually resampled to a common space);
    failing loudly is the safer behavior to keep, not a regression to work
    around.

    Args:
        images: Scalar images to merge, all with the same size/geometry.

    Returns:
        The merged multi-component image.

    Raises:
        RuntimeError: If the images don't all share the same geometry (from
            the underlying sitk.Compose call).
    """
    return sitk.Compose(list(images))


def new_image_like(reference: sitk.Image, array: npt.NDArray[Any]) -> sitk.Image:
    """Build a scalar image with new pixel data but a reference's geometry.

    Replacement for the ANTsImage instance method `.new_image_like(array)`
    (not a top-level ants.* function, but used the same way MIST uses the
    other functions in this module -- e.g. after an in-place transform on a
    mask's array, to get back an image with the mask's original spacing/
    origin/direction). `array` may have a different dtype than `reference`
    (ants.new_image_like doesn't force a dtype either -- the new image just
    takes on the array's own dtype); it must have the same shape.

    Args:
        reference: Image to copy spacing/origin/direction from.
        array: New pixel data, in (x, y, z) axis order, matching
            `reference`'s size.

    Returns:
        A new image with `array`'s data and `reference`'s geometry.

    Raises:
        RuntimeError: If `array`'s shape doesn't match `reference`'s size
            (from the underlying sitk.Image.CopyInformation call; ants'
            equivalent check raises ValueError instead, a difference in
            exception type only -- both fail loudly on a mismatch).
    """
    new_image = sitk.GetImageFromArray(array.T)
    new_image.CopyInformation(reference)
    return new_image
