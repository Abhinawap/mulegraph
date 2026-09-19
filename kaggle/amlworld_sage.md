# Running the AMLworld grid on Kaggle

The two GraphSAGE rows on HI-Small need more host memory than a 7 GB laptop has for the
10M-edge undirected graph plus PyG's sampler. A Kaggle notebook with a P100 (16 GB GPU, ~29 GB
RAM) runs the whole `configs/amlworld_hi_small.yaml` grid in one go.

Notebook settings: Accelerator **GPU P100**, Internet **on**, add the dataset
`ealtman2019/ibm-transactions-for-anti-money-laundering-aml` as input.

Cell 1 — install the repo (pinned deps; ~3 min):

```bash
%%bash
pip install -q uv
git clone --depth 1 https://github.com/Abhinawap/mulegraph.git /kaggle/working/mulegraph
cd /kaggle/working/mulegraph
uv sync --frozen 2>&1 | tail -2
```

Cell 2 — point the loader at the mounted CSV and run (XGB rows in minutes, SAGE rows ~30–40 min
each with `epochs: 10`; 3 seeds × 4 configs). `MPLBACKEND=Agg` overrides Kaggle's inline backend,
which the uv environment does not have:

```bash
%%bash
cd /kaggle/working/mulegraph
src=$(find /kaggle/input -name HI-Small_Trans.csv | head -1)   # mount path varies by notebook
[ -n "$src" ] || { echo "HI-Small_Trans.csv not under /kaggle/input: add the dataset in Input"; exit 1; }
echo "using $src"
mkdir -p data/raw/amlworld/hi_small
ln -sf "$src" data/raw/amlworld/hi_small/HI-Small_Trans.csv
MPLBACKEND=Agg uv run mulegraph run --config configs/amlworld_hi_small.yaml 2>&1 | grep --line-buffered -v "gfp t="
```

Cell 3 — bundle the outputs for download (Output tab → `amlworld_run.zip`):

```bash
%%bash
cd /kaggle/working/mulegraph
zip -qr /kaggle/working/amlworld_run.zip report/tables report/figures mlruns/mlflow.db
ls -la /kaggle/working/amlworld_run.zip
```

Back on the laptop, unzip somewhere outside the repo, copy `report/tables/amlworld_hi_small_*`
and `report/figures/amlworld_hi_small_curves.png` into `report/`, keep the Kaggle `mlflow.db`
as `mlruns/kaggle_amlworld.db`, and paste the four rows of
`report/tables/amlworld_hi_small_results_wide.md` into the README table. The MLflow CSV export
for `report/exports/` comes from that database:

```bash
uv run python -c "
import mlflow; mlflow.set_tracking_uri('sqlite:///mlruns/kaggle_amlworld.db')
mlflow.search_runs(experiment_names=['amlworld_hi_small']).to_csv('report/exports/amlworld_hi_small_runs.csv', index=False)"
```

If a session dies mid-grid, the graph and feature caches under `data/cache/` are gone with it;
re-running the notebook rebuilds them (~10 min) before the fits.
