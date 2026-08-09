"""Project-wide controls for deterministic stochastic computations."""

RANDOM_SEED = 0

# One canonical low-fidelity subsample shared by every model and every CV. All
# models draw from the same LF pool (XLF_10k) via mf_gp.select_lf_subsample, so
# fixing both the size and the seed makes them select the identical rows. The
# canonical subsample is fully defined by (RANDOM_SEED, LF_SUBSAMPLE_SIZE).
LF_SUBSAMPLE_SIZE = 200
