# RefineBridge

**ICASSP 2026** · 2.3M parameters · 11–71% MSE reduction across 81/90 experimental configurations · Consistently outperforms LoRA fine-tuning on S&P 500, WTI crude oil, and EURUSD.

## Pipeline overview

The full pipeline is four steps:

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/generate_dataset.py` | Runs TSFM inference over your price data, saves `.npy` triplets |
| 2 | `scripts/train.py` | Trains RefineBridge on the generated triplets |
| 3 | `scripts/hyperparam_search.py` | Grid searches `temperature × n_timesteps` to find the best inference config |
| 4 | `scripts/evaluate.py` | Evaluates with the best config, computes all metrics and saves figures |

Steps 3 and 4 are separate so you can run the search once and evaluate many times cheaply.

---

## Project structure

```
RefineBridge/
│
├── data/                        ← generated datasets       (git-ignored)
├── checkpoints/                 ← training outputs         (git-ignored)
├── results/                     ← evaluation outputs       (git-ignored)
│
├── src/
│   ├── data/
│   │   ├── tsfm.py              TSFM inference wrappers (Chronos, Moirai, TimeMoE)
│   │   └── generate.py          sliding-window triplet builder → .npy
│   ├── models/
│   │   ├── noise_scheduler.py   forward diffusion + gmax/vp/sb/constant schedules
│   │   ├── unet.py              BridgeUNet — 1D U-Net score estimator
│   │   ├── context_encoder.py   ContextEncoder — DLinear decomposition
│   │   └── refinebridge.py      BridgeSDE + RefineBridge (public API)
│   ├── dataset.py               RefineBridgeDataset — loads .npy, normalises
│   ├── training.py              train_model, checkpointing, loss curves
│   └── evaluate.py              all metrics, plots, inference timing
│
├── scripts/
│   ├── generate_dataset.py      Step 1
│   ├── train.py                 Step 2
│   ├── hyperparam_search.py     Step 3
│   └── evaluate.py              Step 4
│
├── requirements.txt
└── README.md
```

---

## Environment setup

**Requirements:** Python 3.10+, PyTorch 2.1+

```bash
git clone https://github.com/your-org/RefineBridge.git
cd RefineBridge
```

```bash
conda create -n refinebridge python=3.10
conda activate refinebridge
```

```bash
pip install -r requirements.txt
```

### Foundation model dependencies

RefineBridge itself has no dependency on any specific TSFM. You only need to install the one you plan to use for data generation. Each has its own setup requirements — refer to the official repos:

**Chronos** — [github.com/amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
```bash
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

**Moirai** — [github.com/SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts)
```bash
pip install uni2ts
```

**TimeMoE** — [github.com/Time-MoE/Time-MoE](https://github.com/Time-MoE/Time-MoE)
```bash
pip install transformers  # TimeMoE loads via HuggingFace with trust_remote_code=True
```

> Each of these models has its own hardware requirements, model weights, and occasionally additional dependencies. Check their respective GitHub pages before running for the first time, particularly if you are on a machine without a GPU.

Verify the core package is importable before continuing:

```bash
python -c "from src.models.refinebridge import RefineBridge; print('OK')"
```

---

## Step 1 — Generate dataset

This step runs your chosen foundation model over raw price data using a sliding window and saves `(context, prediction, ground_truth)` triplets as `.npy` files. These triplets are what RefineBridge trains on.

### Input format

A CSV file with a date column and a price column, ordered chronologically:

```
Date,Close
2010-01-04,1132.99
2010-01-05,1136.52
2010-01-06,1137.14
...
```

### Run

```bash
python scripts/generate_dataset.py \
    --input      data/raw/SNP500.csv \
    --value-col  Close \
    --date-col   Date \
    --asset      SNP500 \
    --pred-len   21 \
    --context-len 252 \
    --tsfm       chronos \
    --device     cuda \
    --output-dir data/CHRONOS_bridge_dataset
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Path to your price CSV |
| `--value-col` | required | Column name containing prices |
| `--asset` | required | Name used in the output folder |
| `--pred-len` | `21` | Prediction horizon H (days) |
| `--context-len` | `252` | Context window fed to the TSFM (252 ≈ 1 trading year) |
| `--tsfm` | `chronos` | Foundation model: `chronos`, `moirai`, or `timemoe` |
| `--model` | auto | Override the default HuggingFace model name for your chosen TSFM |
| `--device` | `cuda` | `cuda`, `mps`, or `cpu` |

### Default model names per TSFM

| `--tsfm` | Default `--model` | Alternatives |
|---|---|---|
| `chronos` | `amazon/chronos-t5-large` | `amazon/chronos-t5-small`, `amazon/chronos-t5-base` |
| `moirai` | `Salesforce/moirai-1.0-R-large` | `Salesforce/moirai-1.0-R-small`, `Salesforce/moirai-1.0-R-base` |
| `timemoe` | `Maple728/TimeMoE-50M` | `Maple728/TimeMoE-200M-Dense`, `Maple728/TimeMoE-200M` |

### Output

```
data/CHRONOS_bridge_dataset/SNP500/SNP500_price_21/
├── train.npy
├── val.npy
├── test.npy
└── dataset_summary.txt
```

Each `.npy` file contains an array of sample dicts with keys:
`context_window`, `prediction`, `ground_truth`, `stats`, `entity_id`, `variable`, `id`

The `stats` dict carries the per-sample normalisation parameters (`gt_mean`, `gt_std`, `pred_mean`, `pred_std`, `context_mean`, `context_std`) needed to denormalise predictions back to price space at evaluation time.

---

## Step 2 — Train

Open `scripts/train.py` and edit the CONFIG block at the top:

```python
TRAIN_DATA_PATH = "data/CHRONOS_bridge_dataset/SNP500/SNP500_price_21/train.npy"
VAL_DATA_PATH   = "data/CHRONOS_bridge_dataset/SNP500/SNP500_price_21/val.npy"
OUTPUT_DIR      = "checkpoints/SNP500_price_21"

CONTEXT_SEQ_LEN = 252
PRED_SEQ_LEN    = 21      # must match --pred-len used in Step 1

BETA_MIN        = 0.01    # default noise schedule works well as a starting point
BETA_MAX        = 50.0

NUM_EPOCHS      = 10000
BATCH_SIZE      = 16
```

Run:

```bash
python scripts/train.py
```

Training saves checkpoints automatically:

```
checkpoints/SNP500_price_21/
├── sb_model_best_V2.pt              ← best val loss — use this for evaluation
├── sb_model_final_E5000_V2.pt       ← stamped copies of the top-5 best epochs
├── sb_model_epoch_500_V2.pt         ← periodic saves every 500 epochs
├── training_validation_losses_V2.png
└── training_validation_losses_log_V2.png
```

Early stopping is enabled by default. To run the full `NUM_EPOCHS` regardless, set `USE_EARLY_STOPPING = False`.

### Noise schedule

The paper found that two noise schedule settings work best depending on the prediction horizon:

| Horizon H | `BETA_MIN` | `BETA_MAX` | Why |
|---|---|---|---|
| 5, 10 days | `0.01` | `50.0` | Aggressive noise for rapid short-horizon refinement |
| 21, 63, 126 days | `0.0001` | `0.02` | Gentle noise for fine long-horizon corrections |

The defaults (`0.01`, `50.0`) are a safe starting point for any horizon. Tune if needed after running the hyperparam search.

---

## Step 3 — Hyperparameter search

Before running final evaluation, find the best `temperature` and `n_timesteps` for inference. These control how stochastic the refinement is and how many diffusion steps are taken — they have a significant effect on quality and do not require retraining.

Open `scripts/hyperparam_search.py` and set the paths:

```python
TRAIN_DATA_PATH = "data/CHRONOS_bridge_dataset/SNP500/SNP500_price_21/train.npy"
TEST_DATA_PATH  = "data/CHRONOS_bridge_dataset/SNP500/SNP500_price_21/test.npy"
CHECKPOINT_PATH = "checkpoints/SNP500_price_21/sb_model_best_V2.pt"
OUTPUT_DIR      = "results/hyperparam_search/SNP500_price_21"

# Grid to sweep — these defaults match the paper
TEMPERATURES = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 10000]
N_TIMESTEPS  = [1, 10, 50, 100, 1000]
```

Run:

```bash
python scripts/hyperparam_search.py
```

This sweeps all 35 combinations (7 temperatures × 5 step counts) and saves:

```
results/hyperparam_search/SNP500_price_21/
├── all_results.txt              ← full metrics for every combination
├── summary_results.csv          ← one row per config, importable into pandas/Excel
└── mse_improvement_heatmap.csv  ← pivot table: n_timesteps × temperature
```

At the end, the best config for each metric is printed:

```
  Best raw MSE              temp=0.01    steps=100    mse_refined=0.002187
  Best MSE improvement %    temp=0.01    steps=1000   mse_improvement=51.2%
  Best IC                   temp=0.5     steps=50     ic_refined=0.3102
  Best directional acc.     temp=1.0     steps=100    dir_acc=59.1%
```

**Take the `temperature` and `n_timesteps` that give the best MSE improvement and plug them into Step 4.**

---

## Step 4 — Evaluate

Open `scripts/evaluate.py`, set the paths, and plug in the best `TEMPERATURE` and `N_TIMESTEPS` from Step 3:

```python
TRAIN_DATA_PATH = "data/CHRONOS_bridge_dataset/SNP500/SNP500_price_21/train.npy"
TEST_DATA_PATH  = "data/CHRONOS_bridge_dataset/SNP500/SNP500_price_21/test.npy"
CHECKPOINT_PATH = "checkpoints/SNP500_price_21/sb_model_best_V2.pt"
OUTPUT_DIR      = "results/SNP500_price_21"

SAMPLER         = "sde"
N_TIMESTEPS     = 100     # ← from Step 3
TEMPERATURE     = 0.01    # ← from Step 3
```

Run:

```bash
python scripts/evaluate.py
```

### Console output

Metrics are reported across three normalisation regimes to give a complete picture:

```
============================================================
  Evaluation Results — SDE sampler
============================================================
  Steps       : 100
  Temperature : 0.01
  Samples     : 1248

  -- Raw metrics --
  MSE     orig=0.004231  ref=0.002187  (-48.3% down)
  MAE     orig=0.047821  ref=0.031204  (-34.7% down)

  -- Z-norm metrics --
  MSE     orig=0.312100  ref=0.187400  (-40.0% down)
  MAE     orig=0.412300  ref=0.279100  (-32.3% down)

  -- Robust (Median-MAD) metrics --
  Robust median   4521.3
  Robust MAD         87.4
  MSE     orig=0.003910  ref=0.002050  (-47.6% down)
  MAE     orig=0.044200  ref=0.029800  (-32.6% down)

  -- Ranking metrics --
  IC          orig=0.1821  ref=0.2943  (delta=+0.1122)
  ICIR        orig=1.8210  ref=2.9430  (delta=+1.1220)
  Rank IC     orig=0.1654  ref=0.2731  (delta=+0.1077)
  Rank ICIR   orig=1.6540  ref=2.7310  (delta=+1.0770)

  -- Directional accuracy --
  Foundation model    52.3%
  RefineBridge        57.8%
```

### Figures saved

```
results/SNP500_price_21/
├── sample_predictions_sde.png               ← random qualitative samples
├── top_mse/
│   ├── top_mse_1_SNP500_sde.png             ← individual top-performer plots
│   ├── top20_mse_performers_SNP500_sde.png  ← combined 20-panel grid
│   └── top20_mse_improvement_summary_SNP500_sde.png
└── top_mae/
    └── ...
```

---

## Running on your own asset

The pipeline works on any univariate price series. To run on a new asset:

1. Prepare a CSV with a date column and a single price column, ordered chronologically.
2. Run Step 1 with `--asset YOUR_ASSET --pred-len YOUR_HORIZON`.
3. Set `PRED_SEQ_LEN` in `scripts/train.py` to match your chosen horizon.
4. Run Steps 2–4.

A separate RefineBridge model should be trained per asset and per horizon. The paper trained and evaluated independent models for each (asset, horizon, TSFM) combination — 90 configurations in total.

### Running all three TSFMs on the same asset

To replicate the paper's comparison between Chronos, Moirai, and TimeMoE, run Step 1 three times with different `--tsfm` flags and different `--output-dir` folders, then train a separate model on each:

```bash
# Generate with Chronos
python scripts/generate_dataset.py --input data/raw/SNP500.csv --value-col Close \
    --asset SNP500 --pred-len 21 --tsfm chronos \
    --output-dir data/CHRONOS_bridge_dataset

# Generate with Moirai
python scripts/generate_dataset.py --input data/raw/SNP500.csv --value-col Close \
    --asset SNP500 --pred-len 21 --tsfm moirai \
    --output-dir data/MOIRAI_bridge_dataset

# Generate with TimeMoE
python scripts/generate_dataset.py --input data/raw/SNP500.csv --value-col Close \
    --asset SNP500 --pred-len 21 --tsfm timemoe \
    --output-dir data/TIMEMOE_bridge_dataset
```

Each model then trains on its own dataset and refines that specific TSFM's predictions.

---

## Citation

```bibtex
@inproceedings{refinebridge2026,
  title     = {RefineBridge: Improving Financial Time Series Foundation Model
               Predictions via Schrödinger Bridge},
  booktitle = {Proceedings of ICASSP},
  year      = {2026}
}
```

---

## License

MIT