#!/usr/bin/env bash
# PRESSURE-BROADENING STRESS TEST -- generation runner (Kawahara feedback).
#
# Question: the npe_v7_opmarg marginalized model was trained over a prior that
# perturbs opacity with (a) discrete line-list swaps and (b) a smooth
# multiplicative cross-section field. NEITHER changes a line's WIDTH. Kawahara
# reports that pressure broadening (HITEMP-Air vs HITEMP-H2: half-width AND
# temperature exponent) was the dominant systematic in his group's Luhman 16
# analysis. Does the marginalized posterior stay calibrated on spectra whose
# broadening is wrong in a way the prior never saw?
#
# MECHANISM (see sculptor/forward_model/opacity.py:apply_broadening_perturbation)
# The raw g395h line lists were DELETED from GCS, so the build-time
# SCULPTOR_GAMMA_SCALE_<MOL> hook (which edits mdb.gamma_air before OpaPremodit
# construction) can no longer be exercised for this track, and it never covered
# ExoMol providers or the temperature exponent. Instead we perturb the SAVED
# PreMODIT operator: PreMODIT evaluates every line width from two lookup grids,
#     ngamma(T,P) = ngamma_ref_grid[i] * (T/Tref_broadening)**(-n_Texp_grid[j]) * P
# so scaling ngamma_ref_grid by s and shifting n_Texp_grid by d is EXACTLY a
# uniform width scale + temperature-exponent shift on every line, with the same
# grid quantization as the unperturbed operator. No rebuild, no line lists.
#
# BASELINE. The generated set matches the ablation stage-1 "smooth_only" config
# bit-for-bit except for broadening (same seed 413612, 512 spectra, 8x64 shards,
# production v1 smooth-field priors, SCULPTOR_FORCE_COMBO=0 pinning the base
# line lists). outputs/paper_opmarg/ablation/stage1_eval.json already carries the
# matched no-broadening-perturbation coverage for the same models, so the
# comparison is strictly one-axis.
#
# WHY HOST CPU. stars-stable cannot form an 8-host TPU mesh right now (workers
# 0-1 wedged, 2/5/6 occupied by another project), and jax.distributed.initialize
# blocks forever without all 8. run_generation_spmd.py --process-index/--process-count
# skips the distributed init and splits the same shard space over a worker
# SUBSET; JAX_PLATFORMS=cpu runs the identical jitted/vmapped forward model on
# the 240-vCPU host. Shard assignment, Halton replay and the GCS rendezvous are
# unchanged, so the union of the subset's shards equals a full SPMD run.
#
# ENV
#   MODE=smoke|gen        smoke: tiny local OFF-vs-ON pair + timing (default)
#   SET=<name>            output subdir under $OUT_BASE (gen mode)
#   GAMMA=<float>         SCULPTOR_BROADENING_GAMMA_SCALE  (1.0 = off)
#   TEXP=<float>          SCULPTOR_BROADENING_TEXP_SHIFT   (0.0 = off)
#   RANK/WORLD            manual SPMD rank / world size (gen mode)
#   NSPEC/SHARD_SIZE/SEED defaults 512 / 64 / 413612
#   DETACH=1              setsid nohup the generation (gen mode)
#
# Monitor via GCS, never SSH polling:
#   gs://sculptor-research-c2/spectra/g395h/broadening_test/<SET>/done_spmd/worker_*.json
set -u
log(){ echo "[$(date -u +%H:%M:%S) $(hostname)] $*"; }
GCS=gs://sculptor-research-c2
MODE="${MODE:-smoke}"
SET="${SET:-}"
GAMMA="${GAMMA:-1.0}"
TEXP="${TEXP:-0.0}"
SEED="${SEED:-413612}"
NSPEC="${NSPEC:-512}"
SHARD_SIZE="${SHARD_SIZE:-64}"
DEVICE_BATCH="${DEVICE_BATCH:-2}"
RANK="${RANK:-0}"
WORLD="${WORLD:-1}"
DETACH="${DETACH:-0}"
DISK_MIN_GB="${DISK_MIN_GB:-12}"
CODE=${CODE:-$HOME/exoJAX-amortized-broadening}
OUT_BASE="${OUT_BASE_GCS:-$GCS/spectra/g395h/broadening_test}"
PRIORS_V1_GCS="${PRIORS_V1_GCS:-$GCS/deps/interlist_ratio_priors_v1.json}"

disk_guard(){
  local free_gb
  free_gb=$(df -BG "$HOME" | awk 'NR==2{gsub(/G/,"",$4); print $4}')
  log "disk free at \$HOME = ${free_gb} GB (min ${DISK_MIN_GB})"
  if [ -z "$free_gb" ] || [ "$free_gb" -lt "$DISK_MIN_GB" ]; then
    log "FATAL: disk free ${free_gb} GB < ${DISK_MIN_GB} GB -- aborting"; exit 1
  fi
}

# The CPU forward model peaks at ~310 GB resident (measured: 8-spectrum smoke,
# device-batch 2, max RSS 308.9 GB). Several pod workers host another project's
# long-running job holding ~115 GB, which leaves them ~280 GB -- launching there
# would OOM and take that job down with us. Refuse to start without headroom.
MEM_MIN_GB="${MEM_MIN_GB:-340}"
mem_guard(){
  local avail_gb
  avail_gb=$(free -g | awk '/^Mem:/{print $7}')
  log "memory available = ${avail_gb} GB (min ${MEM_MIN_GB}; forward model peaks ~310 GB)"
  if [ -z "$avail_gb" ] || [ "$avail_gb" -lt "$MEM_MIN_GB" ]; then
    log "FATAL: available memory ${avail_gb} GB < ${MEM_MIN_GB} GB -- aborting (would OOM)"
    exit 1
  fi
}

log "BROADENING STRESS: mode=$MODE set=${SET:-<none>} gamma=$GAMMA texp=$TEXP rank=$RANK/$WORLD"
log "STEP 0: clean stale gen dirs"
rm -rf "$HOME"/_spmd_gen_w* "$CODE"/_spmd_gen_w* 2>/dev/null
disk_guard
mem_guard

log "STEP 1: refresh code"
gcloud storage cp $GCS/deps/broadening_code.tar.gz /tmp/broadening_code.tar.gz 2>&1 | tail -1
rm -rf "$CODE"; mkdir -p "$CODE"; tar xzf /tmp/broadening_code.tar.gz -C "$CODE"
cd "$CODE"
# Dependency-free TEXT gates: a staged tarball without these would silently
# generate an UNPERTURBED set (or die on the unknown CLI flags).
grep -q 'def apply_broadening_perturbation' sculptor/forward_model/opacity.py \
  && grep -q '_apply_broadening_perturbations' sculptor/forward_model/opacity.py \
  || { log "FATAL: staged code lacks the broadening-perturbation hook"; exit 1; }
grep -q 'manual_rank' scripts/run_generation_spmd.py \
  || { log "FATAL: staged run_generation_spmd.py lacks --process-index support"; exit 1; }
log "code gates passed"

log "STEP 1.5: python environment (idempotent)"
python3 -c "import numpy, jax, exojax" 2>/dev/null || {
  gcloud storage cp $GCS/deps/tmp_setup_worker_c2.sh /tmp/setup_worker.sh 2>&1 | tail -1
  [ -f /tmp/setup_worker.sh ] || { log "FATAL: env setup script missing on GCS"; exit 1; }
  bash /tmp/setup_worker.sh >> "$HOME/broadening_env_setup.log" 2>&1
  python3 -c "import numpy, jax, exojax; print('env ok')" \
    || { log "FATAL: env setup failed (see broadening_env_setup.log)"; exit 1; }
}

log "STEP 2: stage opacity caches + fastchem + CIA + dispersion (idempotent, ~5 GB)"
mkdir -p /tmp/opa_cache_alt_h2o /tmp/opa_cache_alt_ch4 /tmp/exojax_db
[ -f /tmp/opa_cache_g395h_v4_h2opok/H2O_opa_premodit.npz ] || \
  gcloud storage cp -r "$GCS/opa_cache_g395h_v4_h2opok" /tmp/ 2>&1 | tail -1
[ -f /tmp/opa_cache_alt_h2o/H2O_opa_premodit.npz ] || \
  gcloud storage cp \
    "$GCS/sequential/round_2/evaluation/opa_cache_g395h_v3/H2O_opa_premodit.npz" \
    "$GCS/sequential/round_2/evaluation/opa_cache_g395h_v3/H2O_opa_premodit_metadata.json" \
    /tmp/opa_cache_alt_h2o/ 2>&1 | tail -1
[ -f /tmp/opa_cache_alt_ch4/CH4_opa_premodit.npz ] || \
  gcloud storage cp "$GCS/opa_cache_g395h_v5_ch4mm/CH4_opa_premodit.npz" \
    "$GCS/opa_cache_g395h_v5_ch4mm/CH4_opa_premodit_metadata.json" \
    /tmp/opa_cache_alt_ch4/ 2>&1 | tail -1
[ -f /tmp/exojax_db/fastchem_grid.h5 ] || \
  gcloud storage cp "$GCS/exojax_db/fastchem/fastchem_grid.h5" /tmp/exojax_db/fastchem_grid.h5 2>&1 | tail -1
[ -d /tmp/exojax_db/CIA ] || gcloud storage cp -r "$GCS/exojax_db/CIA" /tmp/exojax_db/ 2>&1 | tail -1
[ -f "$CODE/tmp_g395h_disp_reference.fits" ] || \
  gcloud storage cp "$GCS/sequential/round_2/evaluation/tmp_g395h_disp_reference.fits" "$CODE/" 2>&1 | tail -1

log "verify staging"
for f in /tmp/opa_cache_g395h_v4_h2opok/H2O_opa_premodit.npz \
         /tmp/opa_cache_g395h_v4_h2opok/CO2_opa_premodit.npz \
         /tmp/opa_cache_alt_h2o/H2O_opa_premodit.npz \
         /tmp/opa_cache_alt_ch4/CH4_opa_premodit.npz \
         /tmp/exojax_db/fastchem_grid.h5 "$CODE/tmp_g395h_disp_reference.fits"; do
  [ -f "$f" ] || { log "FATAL: missing $f"; exit 1; }
done
[ -d /tmp/exojax_db/CIA ] || { log "FATAL: missing /tmp/exojax_db/CIA"; exit 1; }
disk_guard

log "STEP 3: production smooth-field priors"
PRIORS_LOCAL=/tmp/broadening_priors_v1.json
gcloud storage cp "$PRIORS_V1_GCS" "$PRIORS_LOCAL" 2>&1 | tail -1
[ -f "$PRIORS_LOCAL" ] || { log "FATAL: v1 priors missing at $PRIORS_V1_GCS"; exit 1; }

# ---- generation environment: IDENTICAL to ablation smooth_only, plus broadening ----
export EXOJAX_ROOT=/tmp/exojax_db PYTHONUNBUFFERED=1 SCULPTOR_DOWNLOAD_BROADENING=0
export SCULPTOR_OPA_CACHE_DIR=/tmp/opa_cache_g395h_v4_h2opok SCULPTOR_READ_OPA_CACHE=1
export SCULPTOR_CLOUD_MODEL="${CLOUD_MODEL:-ackerman_marley}"
export SCULPTOR_OPACITY_PERTURB=1
export SCULPTOR_OPACITY_PERTURB_MODES=4
export SCULPTOR_OPACITY_PERTURB_PRIORS="$PRIORS_LOCAL"
export SCULPTOR_OPA_ALT_CACHE_DIR_H2O=/tmp/opa_cache_alt_h2o
export SCULPTOR_OPA_ALT_CACHE_DIR_CH4=/tmp/opa_cache_alt_ch4
export SCULPTOR_FORCE_COMBO="${FORCE_COMBO:-0}"
# CPU generation: no TPU mesh available on this pod (see header).
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export SPMD_RENDEZVOUS_TIMEOUT_S="${SPMD_RENDEZVOUS_TIMEOUT_S:-86400}"

LOGDIR=/tmp/broadening_logs; mkdir -p "$LOGDIR"

if [ "$MODE" = "smoke" ]; then
  SM_NSPEC="${SM_NSPEC:-8}"; SM_SHARD="${SM_SHARD:-4}"
  for phase in off on; do
    outdir=/tmp/broadening_smoke_$phase
    rm -rf "$outdir"; mkdir -p "$outdir"
    if [ "$phase" = "off" ]; then
      unset SCULPTOR_BROADENING_GAMMA_SCALE SCULPTOR_BROADENING_TEXP_SHIFT
    else
      export SCULPTOR_BROADENING_GAMMA_SCALE="$GAMMA" SCULPTOR_BROADENING_TEXP_SHIFT="$TEXP"
    fi
    log "SMOKE $phase: $SM_NSPEC spectra, shard $SM_SHARD, CPU"
    /usr/bin/time -v python3 scripts/run_generation_spmd.py --track g395h \
      --num-spectra "$SM_NSPEC" --shard-size "$SM_SHARD" --seed "$SEED" \
      --device-batch-size "$DEVICE_BATCH" --process-index 0 --process-count 1 \
      --fastchem-path /tmp/exojax_db/fastchem_grid.h5 \
      --output "$outdir" > "$LOGDIR/smoke_$phase.log" 2>&1
    rc=$?
    log "  smoke $phase rc=$rc"
    grep -E 'broadening_perturbation_applied|broadening_perturbed|shard_written|Maximum resident|Elapsed' \
      "$LOGDIR/smoke_$phase.log" | tail -8
    [ $rc -eq 0 ] || { tail -30 "$LOGDIR/smoke_$phase.log"; log "FATAL: smoke $phase failed"; exit $rc; }
  done
  log "SMOKE COMPARE (off vs on)"
  python3 - /tmp/broadening_smoke_off /tmp/broadening_smoke_on <<'PY'
import sys, h5py, numpy as np
from pathlib import Path
off, on = (Path(p) for p in sys.argv[1:3])
so = sorted(off.glob("shard_*.h5")); sn = sorted(on.glob("shard_*.h5"))
assert so and len(so) == len(sn), (so, sn)
A = np.concatenate([h5py.File(p)["spectra_clean"][...] for p in so]).astype(np.float64)
B = np.concatenate([h5py.File(p)["spectra_clean"][...] for p in sn]).astype(np.float64)
PA = np.concatenate([h5py.File(p)["params"][...] for p in so])
PB = np.concatenate([h5py.File(p)["params"][...] for p in sn])
frac = (B - A) / np.where(A != 0, A, np.nan)
print("params identical:", bool(np.array_equal(PA, PB)))
print("n_spectra:", A.shape[0], "n_nu:", A.shape[1])
print("rms fractional change: %.5f" % float(np.sqrt(np.nanmean(frac**2))))
print("median |frac|:        %.5f" % float(np.nanmedian(np.abs(frac))))
print("p99 |frac|:           %.5f" % float(np.nanpercentile(np.abs(frac), 99)))
print("max |frac|:           %.5f" % float(np.nanmax(np.abs(frac))))
print("per-object rms frac:", np.round(np.sqrt(np.nanmean(frac**2, axis=1)), 5).tolist())
PY
  log "SMOKE DONE"
  exit 0
fi

# ---------------------------- gen mode ----------------------------
[ -n "$SET" ] || { log "FATAL: SET required in gen mode"; exit 1; }
OUT="$OUT_BASE/$SET"
if [ "$GAMMA" = "1.0" ] && [ "$TEXP" = "0.0" ]; then
  unset SCULPTOR_BROADENING_GAMMA_SCALE SCULPTOR_BROADENING_TEXP_SHIFT
  log "GEN: broadening perturbation OFF (reference set)"
else
  export SCULPTOR_BROADENING_GAMMA_SCALE="$GAMMA" SCULPTOR_BROADENING_TEXP_SHIFT="$TEXP"
  log "GEN: gamma x $GAMMA, n_Texp + $TEXP"
fi
GENLOG="$LOGDIR/gen_${SET}_r${RANK}.log"
log "STEP 4: generation rank $RANK/$WORLD, $NSPEC spectra, shard $SHARD_SIZE, seed $SEED -> $OUT"

if [ "$DETACH" = "1" ]; then
  ( while true; do
      gcloud storage cp "$GENLOG" "$OUT/logs/worker_${RANK}.log" >/dev/null 2>&1
      sleep 300
    done ) </dev/null >/dev/null 2>&1 &
  setsid nohup python3 scripts/run_generation_spmd.py --track g395h \
    --num-spectra "$NSPEC" --shard-size "$SHARD_SIZE" --seed "$SEED" \
    --device-batch-size "$DEVICE_BATCH" --process-index "$RANK" --process-count "$WORLD" \
    --fastchem-path /tmp/exojax_db/fastchem_grid.h5 \
    --output "$OUT" > "$GENLOG" 2>&1 &
  GEN_PID=$!
  log "detached pid=$GEN_PID; monitor $OUT/done_spmd/worker_*.json on GCS"
  sleep 25
  if kill -0 "$GEN_PID" 2>/dev/null; then
    log "generation alive after 25s; log head:"; head -6 "$GENLOG"
  else
    log "WARNING: generation exited within 25s; tail:"; tail -25 "$GENLOG"
    gcloud storage cp "$GENLOG" "$OUT/logs/early_exit_worker_${RANK}.log" 2>&1 | tail -1
    exit 1
  fi
  log "RANK $RANK LAUNCH DONE ($SET)"
  exit 0
fi

python3 scripts/run_generation_spmd.py --track g395h \
  --num-spectra "$NSPEC" --shard-size "$SHARD_SIZE" --seed "$SEED" \
  --device-batch-size "$DEVICE_BATCH" --process-index "$RANK" --process-count "$WORLD" \
  --fastchem-path /tmp/exojax_db/fastchem_grid.h5 \
  --output "$OUT" > "$GENLOG" 2>&1
rc=$?
log "generation rc=$rc $(tail -1 "$GENLOG" | cut -c1-160)"
gcloud storage cp "$GENLOG" "$OUT/logs/worker_${RANK}.log" >/dev/null 2>&1
[ $rc -eq 0 ] || { tail -30 "$GENLOG"; log "FATAL: $SET generation failed"; exit $rc; }
log "BROADENING GENERATION DONE (rank $RANK, set $SET)"
