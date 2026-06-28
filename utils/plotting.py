"""
Shared matplotlib/seaborn style and plot helper functions.
Call ``apply_style()`` once per notebook before any plotting.
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns

COLOR_PALETTE = sns.color_palette("colorblind")

STYLE = [
    {
        "figure.dpi": 300,
        "figure.figsize": (3.5, 3.5 / 1.618),
        "font.size": 9,
        "image.cmap": "inferno",
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman"] + plt.rcParams["font.serif"],
        "xtick.top": True,
        "xtick.direction": "in",
        "ytick.right": True,
        "ytick.direction": "in",
        "mathtext.fontset": "cm",
    }
]


def apply_style() -> None:
    """
    Apply publication-quality matplotlib/seaborn defaults.

    Args:
        None

    Outputs:
        Modifies global matplotlib rcParams in place.

    Returns:
        None
    """
    plt.style.use(STYLE)
    sns.set_palette(COLOR_PALETTE)


def save_fig(filename: str, figures_dir: str, tight: bool = True) -> None:
    """
    Save the current matplotlib figure to the figures directory.

    Args:
        filename: file name including extension (e.g. ``'coverage.png'``).
        figures_dir: absolute path to the output directory; created if absent.
        tight: whether to call ``tight_layout()`` before saving.

    Outputs:
        Saves a PNG file to ``figures_dir/filename``.

    Returns:
        None
    """
    os.makedirs(figures_dir, exist_ok=True)
    if tight:
        plt.tight_layout()
    path = os.path.join(figures_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved → {path}")


def coverage_bar(labels: list[str], counts: list[int], total: int, title: str = "") -> None:
    """
    Draw a horizontal bar chart showing match coverage per dataset join.

    Args:
        labels: list of join-step labels (y-axis).
        counts: list of matched row counts corresponding to each label.
        total: total number of base rows (denominator for percentage).
        title: optional chart title.

    Outputs:
        Displays the plot inline.

    Returns:
        None
    """
    fig, ax = plt.subplots()
    pcts = [c / total * 100 for c in counts]
    bars = ax.barh(labels, pcts, color=COLOR_PALETTE[0])
    ax.set_xlabel("Match rate (%)")
    ax.set_xlim(0, 100)
    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(
            pct + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{cnt:,} ({pct:.0f}%)",
            va="center",
            fontsize=7,
        )
    if title:
        ax.set_title(title)
    plt.tight_layout()
