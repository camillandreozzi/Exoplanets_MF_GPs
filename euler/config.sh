#!/usr/bin/env bash
# Settings for the local driver (euler/euler.sh). Override any of these in the
# environment, or edit them here once and forget about them.

# How to reach the cluster. The default assumes an ssh alias; see euler/README.md
# for the ~/.ssh/config block that sets one up together with key-based login.
EULER_HOST="${EULER_HOST:-euler}"

# Where the project lives on the cluster. Left unexpanded on purpose: it is
# resolved by the remote shell, so $SCRATCH is the cluster's scratch space and
# not anything on this laptop. Scratch is the right home for this: the run
# writes hundreds of lane files and home directories on Euler are small.
EULER_REMOTE_DIR="${EULER_REMOTE_DIR:-\$SCRATCH/exoplanets_GPs}"

# Python environment built by euler/setup_euler.sh.
EULER_VENV="${EULER_VENV:-\$HOME/venvs/exoplanets}"

# Software stack modules loaded before the venv. Check what is current with
# `module avail python` on a login node if this combination stops resolving.
EULER_MODULES="${EULER_MODULES:-stack/2024-06 gcc/12.2.0 python/3.11.6}"

# Default SLURM resources per array task. A task runs CV_LOO_CPU workers, one
# core each, so cores and workers stay in step.
#
# These defaults are sized for the `public` shareholder group, which caps a
# user at cpu=48 and mem=128G across all running jobs at once. array x cores
# must stay under 48 or the surplus tasks pend forever on QOSMaxCpuPerUser,
# and array x cores x mem under 128G or they pend on QOSMaxMemoryPerUser.
# 12 x 4 = 48 cores at 1g = 48G, which fits both with room to spare.
#
# Check your own limits on a login node with:
#   sacctmgr -n show qos public format=Name,MaxTRESPU%60 -P
# and substitute your group's QOS name if you are not on the public share.
EULER_CORES="${EULER_CORES:-4}"
EULER_MEM_PER_CPU="${EULER_MEM_PER_CPU:-1g}"
EULER_TIME="${EULER_TIME:-120:00:00}"
EULER_ARRAY_SIZE="${EULER_ARRAY_SIZE:-12}"
