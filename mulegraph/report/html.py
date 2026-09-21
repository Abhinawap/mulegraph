"""Static, self-contained HTML page for one scored batch (D5); needs only the health payload."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Alerts shown in the page; the full ranked queue is the CSV beside it.
TOP_ALERTS = 25

#: What each detector compares, and what a flag means in plain words.
DETECTORS = {
    "psi": ("input features", "the largest shift of any one feature"),
    "ks": ("input features", "the share of features whose distribution differs"),
    "conf": ("model scores", "how far the score distribution has moved"),
    "alert": ("model scores", "how much the share of alerts has changed"),
}

FLAGGED, CLEAR = "#d55e00", "#0072b2"

STYLE = """
body{font:16px/1.5 system-ui,sans-serif;max-width:56rem;margin:0 auto;padding:1rem 16px 3rem;
color:#1b1f24}
h1{font-size:1.5rem;margin:1rem 0 .25rem} h2{font-size:1.1rem;margin:2rem 0 .5rem}
.banner{padding:.75rem 1rem;border-radius:6px;border-left:6px solid;margin:1rem 0}
.banner.flag{background:#fdece2;border-color:#d55e00}
.banner.clear{background:#e6f1f8;border-color:#0072b2}
.note{color:#4a5560;font-size:.9rem} img{max-width:100%;height:auto}
.wrap{overflow-x:auto} table{border-collapse:collapse;font-size:.9rem;min-width:100%}
th,td{border-bottom:1px solid #d8dde2;padding:.3rem .6rem;text-align:left;white-space:nowrap}
dl{display:grid;grid-template-columns:max-content 1fr;gap:.2rem 1rem;font-size:.9rem}
dt{color:#4a5560} dd{margin:0;overflow-wrap:anywhere}
"""


def _figure(detectors: list[dict[str, Any]], flagged_names: list[str]) -> str:
    """One panel per detector: today's score beside its flag level, as a base64 PNG."""
    fig, axes = plt.subplots(1, len(detectors), figsize=(2.2 * len(detectors), 2.8), squeeze=False)
    for ax, d in zip(axes[0], detectors, strict=True):
        flagged = d["detector"] in flagged_names
        ax.bar([0, 1], [d["score"], d["flag_at"]], color=[FLAGGED if flagged else CLEAR, "#9aa5b1"])
        ax.set_xticks([0, 1], ["today", "flag at"])
        ax.set_title(d["detector"] + (" (flagged)" if flagged else ""), fontsize=10)
        for x, v in enumerate((d["score"], d["flag_at"])):
            ax.annotate(f"{v:.2f}", (x, v), ha="center", va="bottom", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=0.2)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def write_score_report(path: Path, health: dict[str, Any], alerts: pd.DataFrame) -> Path:
    """Write the page for one health payload and its alert queue; no scripts, no external files."""
    e = html.escape
    fit = health["model_fit"]
    flagged = health["flagged"]
    if flagged:
        banner = (
            f'<div class="banner flag"><strong>Drift flagged by {e(", ".join(flagged))}.</strong> '
            "The scores may not be trustworthy today.</div>"
        )
    else:
        banner = '<div class="banner clear"><strong>No detector flagged this batch.</strong></div>'
    rows = "".join(
        f"<tr><td>{e(d['detector'])}</td><td>{e(DETECTORS[d['detector']][0])}</td>"
        f"<td>{e(DETECTORS[d['detector']][1])}</td><td>{d['score']:.3f}</td>"
        f"<td>{d['flag_at']:.3f}</td><td>{'yes' if d['detector'] in flagged else 'no'}</td></tr>"
        for d in health["detectors"]
    )
    first, last = health["reference_batches"]
    facts = {
        "Scored at commit": health["commit"],
        "Model fitted at commit": fit["commit"],
        "Model": f"{fit['model']} ({fit['device']}, xgboost {fit['xgboost']})",
        "Alert threshold (chosen on validation)": f"{health['threshold']:.4f}",
        "Health reference": f"batches {first} to {last}",
        "Dataset / features / split": (
            f"{fit['dataset_version']} / {fit['feature_version']} / {fit['split_hash']}"
        ),
    }
    shown = (
        alerts.head(TOP_ALERTS)
        .drop(columns="unit")
        .to_html(index=False, escape=True, border=0, float_format=lambda v: f"{v:.4f}")
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Batch {health["batch"]} report</title><style>{STYLE}</style></head><body>
<h1>Batch {health["batch"]}: {health["alerts"]:,} alerts from {health["units_scored"]:,} units</h1>
{banner}
<p class="note">{e(health["note"])}</p>
<h2>Health check</h2>
<img alt="Each detector's score for this batch beside its flag level"
 src="data:image/png;base64,{_figure(health["detectors"], flagged)}">
<div class="wrap"><table><tr><th>Detector</th><th>Compares</th><th>A flag means</th>
<th>Score</th><th>Flag level</th><th>Flagged</th></tr>{rows}</table></div>
<h2>Alert queue</h2>
<p class="note">Top {min(TOP_ALERTS, len(alerts))} of {len(alerts):,}, ranked by score. The last
column is how much each feature pushed the score toward laundering, in log-odds.</p>
<div class="wrap">{shown}</div>
<h2>Provenance</h2>
<dl>{"".join(f"<dt>{e(k)}</dt><dd>{e(str(v))}</dd>" for k, v in facts.items())}</dl>
</body></html>
"""
    path.write_text(page, encoding="utf-8")
    return path
