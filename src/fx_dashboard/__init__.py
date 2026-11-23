"""FX vs macro analysis dashboard."""

import os

# OpenBLAS reserves a buffer per CPU thread when numpy loads; on a machine short of memory
# that allocation fails and kills the server. One thread is plenty for these small
# regressions. Must run before numpy/pandas are imported; setdefault lets users override.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "0.1.0"
