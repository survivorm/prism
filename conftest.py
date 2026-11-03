"""Test-session environment (imported by pytest before any test module).

BLAS/LAPACK threads are pinned to 1 BEFORE numpy/torch are imported — on the
small matrices this suite uses (D ≤ 64, K ≤ 128) a multithreaded BLAS is up to
30× SLOWER than a single thread, and the same effect made the E-40 sweeps look
like "hours of tests" (measured 2026-11-03: SVD of [128,1024] = 38 ms @1 thread,
142 ms @4, 1162 ms @12; two such processes on 12 cores produced load average 57).
Thread count never changes results, only speed, so ``setdefault`` keeps any
explicit override from the environment.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
