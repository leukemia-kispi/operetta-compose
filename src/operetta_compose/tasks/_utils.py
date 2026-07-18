"""Shared helpers for the operetta-compose plate-level tasks."""

from pathlib import Path

# Columns identifying a well. `ngio` uses `row` / `column`; these are the
# canonical names used across the operetta-compose tables.
WELL_COLUMNS = ["row", "column"]

# Case-insensitive aliases accepted in user-provided layout files.
_ROW_ALIASES = {"row"}
_COLUMN_ALIASES = {"column", "col"}


def plate_url_from_zarr_url(zarr_url: str) -> str:
    """Return the plate (``*.zarr``) root for an image ``zarr_url``.

    A non-parallel task receives the list of image URLs (e.g.
    ``/path/plate.zarr/C/3/0``). The plate is the first path component ending
    in ``.zarr``.

    Args:
        zarr_url: Path to an OME-Zarr image inside a plate.

    Returns:
        Path to the plate (``*.zarr``) root.
    """
    parts = Path(zarr_url).parts
    for i, part in enumerate(parts):
        if part.endswith(".zarr"):
            return str(Path(*parts[: i + 1]))
    raise ValueError(f"No `.zarr` component found in zarr_url: {zarr_url}")


def well_key(value) -> str:
    """Normalize a single row/column identifier for well matching.

    ``ngio`` stores plate columns zero-padded (e.g. ``"3"`` becomes ``"03"``).
    This returns a padding-insensitive key so that layout identifiers written as
    ``3``, ``"3"`` or ``"03"`` all match the plate's canonical well, while
    non-numeric identifiers (e.g. the row ``"C"``) are compared as-is.

    Args:
        value: A row or column identifier.

    Returns:
        A canonical string key.
    """
    text = str(value).strip()
    try:
        return str(int(text))
    except ValueError:
        return text


def normalize_well_columns(columns: list[str]) -> tuple[str, str]:
    """Find the row and column headers in a layout, accepting common aliases.

    Args:
        columns: Column names of a user-provided layout table.

    Returns:
        A tuple ``(row_column_name, column_column_name)`` with the original
        header names for the row and column identifiers.
    """
    lower_to_original = {c.lower(): c for c in columns}
    row_match = _ROW_ALIASES & lower_to_original.keys()
    if not row_match:
        raise ValueError("Layout must contain a 'row' column.")
    column_match = _COLUMN_ALIASES & lower_to_original.keys()
    if not column_match:
        raise ValueError("Layout must contain a 'column' or 'col' column.")
    return lower_to_original[row_match.pop()], lower_to_original[column_match.pop()]
