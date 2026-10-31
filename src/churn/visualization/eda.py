"""Plotting helpers for the exploratory data analysis."""
import matplotlib.pyplot as plt
import pandas as pd

from churn.config import TARGET, VISUALS_DIR

CHURN_COLORS = {"No": "#4C72B0", "Yes": "#DD8452"}


def churn_rate_by(df: pd.DataFrame, column: str, order=None) -> pd.DataFrame:
    """Churn rate (%) and customer count for every value of `column`."""
    grouped = df.groupby(column)[TARGET].agg(
        customers="size", churn_rate=lambda s: s.eq("Yes").mean() * 100
    )
    grouped["churn_rate"] = grouped["churn_rate"].round(2)
    return grouped.reindex(order) if order is not None else grouped


def plot_churn_rate(df: pd.DataFrame, column: str, ax=None, order=None, title=None):
    """Bar chart of the churn rate per category, with the overall churn rate as reference."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    rates = churn_rate_by(df, column, order)
    overall = df[TARGET].eq("Yes").mean() * 100

    bars = ax.bar(rates.index.astype(str), rates["churn_rate"], color=CHURN_COLORS["Yes"])
    for bar, rate in zip(bars, rates["churn_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.8, f"{rate:.1f}%", ha="center", fontsize=9)
    ax.axhline(overall, color="gray", linestyle="--", linewidth=1, label=f"Overall ({overall:.1f}%)")
    ax.set_title(title or f"Churn rate by {column}")
    ax.set_xlabel("")
    ax.set_ylabel("Churn rate (%)")
    ax.set_ylim(0, max(rates["churn_rate"].max() * 1.2, overall * 1.2))
    ax.tick_params(axis="x", labelrotation=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    return ax


def save_fig(fig, name: str) -> None:
    """Save a figure into the visuals/ folder."""
    VISUALS_DIR.mkdir(exist_ok=True)
    fig.savefig(VISUALS_DIR / name, dpi=150, bbox_inches="tight")
