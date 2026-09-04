# Amortized opacity marginalization

Code, trained models and results for *Amortized Opacity Marginalization Improves C/O Interval Calibration for Brown-Dwarf Retrievals* (Heraty 2026).

Molecular line lists disagree at a level that shifts retrieved C/O, so a retrieval conditioned on a single list quotes error bars that omit a dominant systematic. This repository randomizes opacities while generating the training set, so a neural posterior estimator learns a posterior already marginalized over that uncertainty. On perturbed held-out spectra a fixed-opacity network covers C/O at 43.3% of the nominal 90% level; a three-seed marginalized ensemble reaches 89.4%.

## Install

```
pip install -e .
```

Requires Python 3.10+, JAX and ExoJAX for spectrum generation, PyTorch and zuko for inference. Generation runs on TPU; training and evaluation run on CPU.

> zuko must be importable. Without it the flow silently falls back to a diagonal Gaussian and every posterior is wrong.

## Reproduce

Training sets and model weights are on the Hugging Face Hub (see Data). To regenerate spectra from scratch instead:

```
python scripts/run_generation_spmd.py --track g395h --num-spectra 1000000 \
  --shard-size 128 --seed 0 --output gs://BUCKET/v7_opmarg
```
with `SCULPTOR_OPACITY_PERTURB=1` and `SCULPTOR_OPACITY_PERTURB_PRIORS` pointing at `configs/interlist_ratio_priors_v1.json`. Unset, generation is bit-identical to the unperturbed control.

Train and evaluate:

```
python scripts/run_training.py --train DATA --output MODEL --track g395h --seed 0
python scripts/opmarg_sbc_compare.py --marg-model MODEL --control-model CONTROL --heldout HELDOUT --out results.json
```

## Layout

- `sculptor/forward_model`: ExoJAX-based emission model, opacity handling, clouds, instrument
- `sculptor/data_generation`: samplers, TPU generation, `opacity_perturbation.py`
- `sculptor/npe`: embedding, flow, training, recalibration, inference
- `scripts`: generation, training and the evaluation scripts behind each table
- `results`: result JSONs for every number in the paper
- `paper`: the paper source
- `tests`: unit tests, including the perturbation and broadening equivalence checks

## Data

Training sets, model weights, priors and held-out evaluation sets: **[huggingface.co/datasets/Kaileh57/opacity-marginalization](https://huggingface.co/datasets/Kaileh57/opacity-marginalization)**

Two matched one-million-spectrum sets, `v7_opmarg` (perturbed) and `v6_cloud` (unperturbed control), plus the trained networks and the held-out sets used for every reported number.

## Citation

```bibtex
@misc{heraty2026opacity,
  title  = {Amortized Opacity Marginalization Improves C/O Interval
            Calibration for Brown-Dwarf Retrievals},
  author = {Heraty, Kellen},
  year   = {2026},
  eprint = {2609.01665},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.IM}
}
```

## License

MIT for the code. Data and model weights on the Hub are CC BY 4.0.

## Acknowledgments

We thank Hajime Kawahara for ExoJAX and for his guidance throughout this project, from its earliest framing to the experiments reported here. Supported with Cloud TPUs from Google's TPU Research Cloud.
