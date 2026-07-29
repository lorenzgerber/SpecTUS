# Fine-tuning on a NIST library edition

This documents the process used to fine-tune the pretrained SpecTUS model on a
licensed NIST GC-EI-MS library (first NIST23, later NIST26), and to serve the
result as an HTTP API. **No NIST-derived data or fine-tuned weights are stored
in this repository** — anyone reproducing this needs their own NIST license.
The actual data/checkpoints produced this way currently live outside this repo
(see "Where the data actually lives" below).

## 1. Obtaining the raw library

NIST distributes its GC-EI-MS library (`mainlib`) in a proprietary format,
normally exported to a plaintext `.msp` file via NIST's own MS Search /
lib2nist tool — which is Windows-only. On OSX/Linux, **OpenChrom** can perform
the same SDF/MSP conversion without needing that tool.

The result is a standard NIST-style `.msp` file: one record per compound, with
`NAME`/`FORMULA`/`MW`/`CASNO`/`SMILES`/`NUM PEAKS` fields followed by `m/z
intensity;` peak lines.

## 2. Regenerating `data/nist/{train,valid,test}.*`

`spectus/utils/prepare_nist_dataset.py` reproduces section 1 of
`notebooks/2_data_preprocessing.ipynb` as a standalone, non-interactive script
(the notebook's other sections — ZINC/RASSP/NEIMS synthetic data generation —
are for pretraining and don't need to be re-run for a fine-tune):

1. Place the raw library at `data/nist/mainlib.msp`.
2. Run:
   ```bash
   micromamba run -n trainSpectus python spectus/utils/prepare_nist_dataset.py
   ```
3. This drops spectra with no usable identifier, reconstructs missing SMILES
   from InChI where possible, canonicalizes/destereochemicalizes every SMILES,
   and splits into `train`/`valid`/`test` (80/10/10, grouped by unique SMILES
   so a molecule's spectra never leak across splits, seed 42) — writing
   `data/nist/{train,valid,test}.{msp,jsonl,smi}`.

The `.smi` files (SMILES only, no spectra) are safe to commit — matching this
repo's existing convention (`!data/nist/{train,valid,test}.smi` in
`.gitignore`) — the `.msp`/`.jsonl` files (which include the licensed spectra)
are not.

## 3. Fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 python spectus/train_spectus.py \
  --config-file configs/finetune_nist_24gb.yaml \
  --checkpoint data/models/SpecTUS_pretrained_only/ \
  --additional-info "_nist26_24gb" \
  --wandb-group "finetune_nist" \
  --additional-tags "nist:from_pretrained:24gb:nist26"
```

- Always start from `data/models/SpecTUS_pretrained_only/` (the freely
  available pretrained checkpoint — see the main README for how to fetch it),
  not from a previous NIST edition's fine-tune, so each edition's fine-tune is
  independent and comparable.
- Give each NIST edition its own `--additional-info` tag (e.g. `_nist23_24gb`,
  `_nist26_24gb`) — this becomes part of the checkpoint output path
  (`checkpoints/<wandb-group>/<additional-info>/`), keeping runs from
  colliding.
- `configs/finetune_nist_24gb.yaml` is sized for a single 24GB GPU (batch size
  4, gradient accumulation 8). `max_steps: 74000` is roughly one epoch over
  the full NIST train split; the best checkpoint is auto-selected via
  `load_best_model_at_end` on `eval_NIST_morgan_tanimoto_simil`.
- For a long run, launch it in a detached `tmux` session so it survives an
  SSH disconnect, e.g. `tmux new -s spectus_finetune -d`.

## 4. Verifying a fine-tuned checkpoint

CLI, single spectrum or a full dataset (`configs/predict_own.yaml`, pointing
`dataset.data_path` at either a single-spectrum `.jsonl` or a full
`data/nist/valid.jsonl`):
```bash
python spectus/predict.py \
  --checkpoint checkpoints/finetune_nist/_nist26_24gb/checkpoint-<best>/ \
  --output-folder predictions \
  --config-file configs/predict_own.yaml
```

Or via the Docker service (see the main README's "Running as a service"
section) — same underlying pipeline, served over HTTP:
```bash
docker build -t spectus-service .
docker run -d -p 8000:8000 \
  -v checkpoints/finetune_nist/_nist26_24gb/checkpoint-<best>:/app/checkpoint:ro \
  spectus-service
```

## Where the data actually lives

The raw library, derived `.msp`/`.jsonl` splits, fine-tuned checkpoints, and
training logs for the NIST23 and NIST26 fine-tunes currently live outside this
repo in a local staging location, pending upload to a private repo or
internal storage:

```
~/Data/SpecTUS_nist_artifacts/
├── data_nist/            # mainlib.msp, {train,valid,test}.{msp,jsonl}
├── checkpoints/          # _nist_24gb/, _nist26_24gb/
├── wandb/
└── predictions/
```

<!-- TODO: update this section with the final internal location once decided
     (private repo or corporate vault). -->
