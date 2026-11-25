"""Lossless compatibility for the one verified historical object-dtype NPY."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile

import numpy as np

OBJECT_DATASET = "0308_Tasla_Stock_Price_2274_5.npy"
OBJECT_SHA256 = "7f86d086699df998c7233c2cd2ee3e95002543ef4adeb8ab87b3be0d398068ea"


@contextmanager
def compatible_input(file):
    file = Path(file)
    if file.name != OBJECT_DATASET:
        yield file
        return
    # Only this reviewed source artifact may be deserialized. Never enable
    # pickle loading for arbitrary user-provided arrays.
    if hashlib.sha256(file.read_bytes()).hexdigest() != OBJECT_SHA256:
        raise ValueError("Historical object array hash differs; inspect this input before conversion")
    original = np.load(file, allow_pickle=True)
    if any(type(value) not in (int, float) for value in original.flat):
        raise ValueError("Historical object array contains a non-numerical value")
    numeric = original.astype(np.float64)
    if not np.isfinite(numeric).all() or not np.equal(original, numeric).all():
        raise ValueError("Object-to-float conversion would change data values")
    if not np.array_equal(original.astype(np.float32), numeric.astype(np.float32)):
        raise ValueError("Conversion changes the original training input")
    with tempfile.TemporaryDirectory(prefix="timeseries-numeric-") as folder:
        # Preserve the original filename and dataset category used by the CLI.
        prepared = Path(folder) / file.parent.name / file.name
        prepared.parent.mkdir()
        np.save(prepared, numeric, allow_pickle=False)
        yield prepared
