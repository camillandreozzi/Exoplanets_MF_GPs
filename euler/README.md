# Running the LOO cross-validation on Euler

Everything is driven from your laptop. You never need to edit
`modelling/cv/cv_loo.py` to change what a run does: the whole configuration
block in that file now also reads `CV_LOO_*` environment variables, and the
scripts here set them.

```
euler/
  euler.sh            the local driver — the only script you run by hand
  config.sh           host, remote directory, module and resource defaults
  setup_euler.sh      builds the Python environment on the cluster (run once)
  submit_loo.sh       submits the job array; euler.sh run calls it for you
  loo_lanes.sbatch    one array task = one stride of the lane list
  loo_assemble.sbatch dependent job that writes the summary CSVs
  check_lanes.py      reports which lanes are still missing
  requirements.txt    pinned Python dependencies
```

## One-time setup

**1. Set up passwordless ssh.** Add to `~/.ssh/config`:

```
Host euler
    HostName euler.ethz.ch
    User YOUR_NETHZ
    IdentityFile ~/.ssh/id_ed25519
```

Then `ssh-copy-id euler` and check that `ssh euler true` returns without a
prompt. If you'd rather not add an alias, set `EULER_HOST` instead:

```bash
export EULER_HOST=your_nethz@euler.ethz.ch
```

**2. Build the cluster environment.**

```bash
./euler/euler.sh setup
```

This syncs the code and the CSVs in `data/` to `$SCRATCH/exoplanets_GPs` on
Euler, creates a virtual environment at `~/venvs/exoplanets`, installs
`euler/requirements.txt`, and prints the package versions plus a row count from
`load_full_data()` so you know the data arrived intact.

If `module load` fails, the software stack has moved on. Check `module avail
python` on a login node and set `EULER_MODULES` in `euler/config.sh` to match.

## Running

```bash
# All three models, both fidelity variants, all 195 wavelengths, all 97 folds
./euler/euler.sh run --models 1,2,3

# One model, the full 10000 low-fidelity rows, more cores and a longer limit
./euler/euler.sh run --models 1 --lf 10000 --cores 24 --time 48:00:00 --array 12

# Multi-fidelity only, on a 15-wavelength subset
./euler/euler.sh run --models 3 --variants mf --wavelengths 15
```

`run` syncs your working tree up first, so local edits are always what runs.

Each invocation submits two jobs: an array whose tasks fit lanes, and one
dependent job that assembles every lane file into the summary CSVs. The
dependency is `afterany`, so the assembly still happens even if some array
tasks hit their time limit — you get partial results rather than nothing.

### Options

| Option | Meaning |
| --- | --- |
| `--models LIST` | `1`, `2`, `3`, a comma list, or `all` |
| `--variants LIST` | `sf`, `mf`, or `both` (default) |
| `--lf N` | Low-fidelity training rows; `none` uses all 10000 |
| `--max-folds N` | Cap the LOO folds — use this for pilot runs |
| `--wavelengths N` | Fit an N-wavelength subset instead of all 195 |
| `--model2-wl N` | Wavelengths sampled per Model 2 training spectrum |
| `--tag NAME` | Names the results directory and the jobs |
| `--results-dir DIR` | Results directory, if you don't want the default |
| `--cores N` | Cores per array task; also the number of worker processes |
| `--mem SIZE` | Memory per core (default `4g`) |
| `--time HH:MM:SS` | Wall-clock limit per array task |
| `--array N` | Number of array tasks |
| `--fresh` | Delete the lane directory before submitting |
| `--dry-run` | Print the `sbatch` commands without submitting |

### Watching a run

```bash
./euler/euler.sh status              # squeue for your jobs
./euler/euler.sh logs                # tail -f the newest job log
./euler/euler.sh check --models 1,3  # which lanes are still missing
```

### What gets synced

`up` and `run` push code and the CSVs in `data/`, and skip `.git/`, `results/`,
`data/lit/` (2.4 GB the models never read), caches and PNGs. The push uses
`rsync --delete`, so files you delete locally disappear on the cluster too —
but excluded paths are protected from that, so **syncing up never touches the
results sitting on the cluster.**

### Getting the results back

```bash
./euler/euler.sh down
```

This rsyncs the cluster's `results/` into your local `results/`, so the
assembled CSVs land exactly where the plotting scripts in `modelling/plot/`
already look for them. It never deletes local files the cluster hasn't seen.
To pull one run only:

```bash
./euler/euler.sh down results/cv/loo_lf10000
```

## How the work is split

Model 1 and Model 3 are scheduled as **wavelength lanes**: one lane fits a
single wavelength across all 97 folds. Model 2 isn't separable by wavelength,
so its lanes are folds. Each lane writes its own CSV as soon as it finishes.

Two consequences worth knowing:

- **Resuming is free.** A job that runs out of time keeps every lane it
  completed. Resubmitting the identical command fits only what's missing, so
  the honest response to an under-estimated `--time` is just to run it again.
- **The array is a stride, not a block.** Array task *i* of *n* takes every
  *n*-th lane. The partition is computed from the full lane list rather than
  the pending one, so it's identical in every task no matter when each starts,
  and no two tasks ever fit the same lane.

## One results directory per training set

The default results directory is `results/cv/loo_lf<N>`, named after the
low-fidelity sample size.

That naming is deliberate. Lane files are named by model and wavelength but not
by training set, so a Model 1 run at `--lf 10000` and a Model 2 run at
`--lf 1000` sharing one directory produce lane files that *look* interchangeable
but were fit against different training data — and `run_config.json` records
only whichever run touched it last. Keeping a directory per LF size makes that
impossible. If you deliberately want several models in one directory, give them
all the same `--lf` and the same `--tag`.

## Sizing a run

Measured on one core, fitting the SF and MF variants of a single wavelength
across 3 folds, then extrapolated to a full 97-fold lane and to all 195 lanes:

| Run | Per fold | Per lane (97 folds) | All lanes | Lanes |
| --- | --- | --- | --- | --- |
| Model 1, `--lf 1000` | 17 s | 28 min | 91 core-h | 195 wavelengths |
| Model 1, `--lf 10000` | 134 s | 3.6 h | 704 core-h | 195 wavelengths |
| Model 3, `--lf 1000` | 78 s | 2.1 h | 410 core-h | 195 wavelengths |

Model 2 isn't in the table: its lanes are folds rather than wavelengths, and
the augmented GP's cost depends on `--model2-wl` as much as on `--lf`. Pilot it.

These came off a laptop core, so treat them as the right order of magnitude
rather than a promise. Two things they do show clearly:

- **`--lf` dominates everything.** Going from 1000 to 10000 low-fidelity rows
  costs 7.7x per fit — close to linear in the training rows, which is what you
  expect: the multi-fidelity fits use a Vecchia approximation with 20 neighbours
  (`gp_approx="vecchia"` in `src/model_1.py`, and `DEFAULT_GP_APPROX_MF` in
  `src/model_3.py`), not a dense Cholesky. That also keeps memory near-linear
  rather than *n²*, so the default `--mem 4g` per core is enough at `--lf 10000`.
  Only Model 3's single-fidelity fits are exact, and those train on the 96 HF
  rows alone, so they stay cheap whatever `--lf` is.
- **Model 3 costs about 4.5x Model 1** at the same `--lf`, because each fold
  tunes the boosted mean before fitting.

### Choosing `--time` and `--array`

Wall clock per array task is roughly

```
lane_time x ceil(lanes / array / cores)
```

So Model 1 at `--lf 10000` with `--array 10 --cores 16`: each task owns about 20
of the 195 lanes, spread over 16 workers, so two lanes deep at 3.6 h each is
about 7.5 hours. `--time 12:00:00` leaves comfortable headroom, and if the
estimate is wrong the lanes that finished are still on disk.

**Pilot first.** Two folds on four wavelengths costs minutes and calibrates
everything above for your actual queue:

```bash
./euler/euler.sh run --models 3 --max-folds 2 --wavelengths 4 \
    --array 1 --cores 4 --time 01:00:00 --tag pilot
./euler/euler.sh logs
```

Multiply the per-lane time in that log by 97/2 for the folds and 195/4 for the
wavelengths.

## Troubleshooting

**`No virtual environment at ...`** — run `./euler/euler.sh setup` first.

**`module load` errors** — the stack changed; see step 2 above.

**A worker raises `Worker LF sample does not match ...`** — the results
directory holds `lf_sample_source_indices.csv` from a run with a different
`--lf`. Use a separate directory, or `--fresh`.

**Some lanes failed** — a failed lane writes no file and is retried by the next
submission. `./euler/euler.sh check` lists them; the array task's log says why
each one raised.

**Jobs pend for a long time** — ask for less. Wall-clock limits of 4 hours or
less reach the short queues on Euler and usually start much sooner; with lane
resuming, several short jobs cost you nothing but a resubmission.


./euler/euler.sh down
CV_LOO_RUN_MODEL1=1 CV_LOO_RUN_MODEL3=1 CV_LOO_RUN_MODEL2=0 \
CV_LOO_LF_SAMPLE_SIZE=10000 CV_LOO_RESULTS_DIR=results/cv/loo_lf10000 \
CV_LOO_ASSEMBLE_ONLY=1 python3 modelling/cv/cv_loo.py
