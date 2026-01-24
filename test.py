import sys

import zarr


def _has_written_chunks(array) -> bool:
    nchunks_initialized = getattr(array, "nchunks_initialized", None)
    try:
        if callable(nchunks_initialized):
            return nchunks_initialized() > 0
        if nchunks_initialized is not None:
            return nchunks_initialized > 0
    except Exception:
        return False

    return False


def print_array_shapes(group, prefix: str = "") -> None:
    for name, array in group.arrays():
        if not _has_written_chunks(array):
            continue
        full_name = f"{prefix}{name}" if prefix else name
        print(f"{full_name}: {array.shape}")

    for name, subgroup in group.groups():
        next_prefix = f"{prefix}{name}/" if prefix else f"{name}/"
        print_array_shapes(subgroup, next_prefix)


# pick filename from first arg
filename = sys.argv[1]

x = zarr.open(filename, mode="r")
print_array_shapes(x)
