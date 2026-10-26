"""Render high-resolution mathematical formula cards and system architecture diagram for the HR & Technical Submission Dossier."""

from __future__ import annotations

import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
ARTIFACT_DIR = Path(r"C:\Users\Asus\.gemini\antigravity\brain\52db3254-3ae6-4998-8237-663a16e992c8")


def _draw_card(
    out_path: Path,
    title: str,
    subtitle: str,
    rows: list[tuple[str, str, str]],
    figsize: tuple[float, float] = (12.8, 4.6),
) -> None:
    """Render a clean, styled mathematical equation reference card as PNG."""
    fig, ax = plt.subplots(figsize=figsize, dpi=220)
    ax.set_xlim(0, 12.8)
    n = len(rows)
    total_h = 1.25 + n * 1.15
    ax.set_ylim(0, total_h)
    ax.axis("off")

    # Outer card background
    outer = FancyBboxPatch(
        (0.08, 0.08),
        12.64,
        total_h - 0.16,
        boxstyle="round,pad=0.06,rounding_size=0.15",
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        linewidth=1.5,
    )
    ax.add_patch(outer)

    # Header bar
    header_h = 0.85
    header = FancyBboxPatch(
        (0.08, total_h - header_h - 0.08),
        12.64,
        header_h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor="#1E3A8A",
        edgecolor="#1E3A8A",
        linewidth=1.0,
    )
    ax.add_patch(header)

    ax.text(
        0.38,
        total_h - 0.38,
        title,
        fontsize=12.5,
        fontweight="bold",
        color="#FFFFFF",
        va="center",
        ha="left",
        family="sans-serif",
    )
    ax.text(
        0.38,
        total_h - 0.69,
        subtitle,
        fontsize=9.2,
        color="#DBEAFE",
        va="center",
        ha="left",
        family="sans-serif",
    )

    y_top = total_h - header_h - 0.24
    for idx, (label, formula, note) in enumerate(rows):
        row_y = y_top - (idx + 1) * 1.08 + 0.12
        row_bg = "#FFFFFF" if idx % 2 == 0 else "#F1F5F9"
        box = FancyBboxPatch(
            (0.24, row_y),
            12.32,
            0.96,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=row_bg,
            edgecolor="#E2E8F0",
            linewidth=1.0,
        )
        ax.add_patch(box)

        # Left label
        ax.text(
            0.44,
            row_y + 0.72,
            label,
            fontsize=9.8,
            fontweight="bold",
            color="#0F172A",
            va="center",
            ha="left",
            family="sans-serif",
        )
        # Right explanatory note
        ax.text(
            12.36,
            row_y + 0.72,
            note,
            fontsize=8.6,
            color="#475569",
            va="center",
            ha="right",
            style="italic",
            family="sans-serif",
        )
        # Math formula rendered via Matplotlib mathtext
        ax.text(
            0.52,
            row_y + 0.30,
            formula,
            fontsize=11.5,
            color="#1E3A8A",
            va="center",
            ha="left",
        )

    plt.tight_layout(pad=0.15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_architecture_diagram(out_path: Path) -> None:
    """Render the 4-stage pipeline architecture diagram as a crisp high-DPI PNG."""
    fig, ax = plt.subplots(figsize=(12.8, 3.6), dpi=220)
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    # Background container
    bg = FancyBboxPatch(
        (0.08, 0.08),
        12.64,
        3.44,
        boxstyle="round,pad=0.05,rounding_size=0.14",
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        linewidth=1.5,
    )
    ax.add_patch(bg)

    ax.text(
        6.4,
        3.18,
        "End-to-End Analytics & Decision Support Architecture (Reproducible Python Pipeline)",
        fontsize=12.0,
        fontweight="bold",
        color="#1E3A8A",
        ha="center",
        va="center",
    )

    stages = [
        (
            0.35,
            "1. Raw UCI Data",
            "#EFF6FF",
            "#1D4ED8",
            [
                "• hour.csv (17,379 rows)",
                "• day.csv (731 rows)",
                "• 2011–2012 Capital Bikeshare",
            ],
        ),
        (
            2.85,
            "2. Quality & Features",
            "#F0FDF4",
            "#15803D",
            [
                "• src/data_validation.py (26 checks)",
                "• src/features.py (Cyclical + Rush)",
                "• Leakage Guard (no casual/reg)",
            ],
        ),
        (
            5.35,
            "3. Statistical Proof",
            "#FEFCE8",
            "#B45309",
            [
                "• src/hypothesis.py",
                "• Welch t-test (t = 17.96)",
                "• Mann-Whitney U + HAC OLS",
            ],
        ),
        (
            7.85,
            "4. Demand & Anomaly ML",
            "#FAF5FF",
            "#6D28D9",
            [
                "• src/train_model.py (HGB Log)",
                "• Test R² = 0.904, MAE = 43.54",
                "• src/anomaly.py (8-fold MAD)",
            ],
        ),
        (
            10.35,
            "5. Ops Prototype",
            "#FFF1F2",
            "#BE123C",
            [
                "• src/query_engine.py (NL Chat)",
                "• app.py (7-Tab Streamlit UI)",
                "• tests/ (24 Automated Pytests)",
            ],
        ),
    ]

    box_w = 2.12
    box_h = 2.35
    box_y = 0.42

    for idx, (x, title, face, edge, bullets) in enumerate(stages):
        card = FancyBboxPatch(
            (x, box_y),
            box_w,
            box_h,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.6,
        )
        ax.add_patch(card)

        # Header band inside stage card
        hdr = FancyBboxPatch(
            (x, box_y + box_h - 0.52),
            box_w,
            0.52,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=edge,
            edgecolor=edge,
            linewidth=1.0,
        )
        ax.add_patch(hdr)

        ax.text(
            x + box_w / 2,
            box_y + box_h - 0.26,
            title,
            fontsize=9.3,
            fontweight="bold",
            color="#FFFFFF",
            ha="center",
            va="center",
        )

        for b_i, line in enumerate(bullets):
            ax.text(
                x + 0.12,
                box_y + box_h - 0.88 - b_i * 0.52,
                line,
                fontsize=8.0,
                color="#1E293B",
                ha="left",
                va="center",
            )

        if idx < len(stages) - 1:
            arrow = FancyArrowPatch(
                (x + box_w + 0.03, box_y + box_h / 2),
                (x + box_w + 0.35, box_y + box_h / 2),
                arrowstyle="Simple,tail_width=2.2,head_width=7,head_length=6",
                color="#475569",
            )
            ax.add_patch(arrow)

    plt.tight_layout(pad=0.15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    # Card 1: Data Quality & Feature Engineering Formulas
    eq1_path = FIG_DIR / "eq_feature_engineering.png"
    _draw_card(
        eq1_path,
        title="Mathematical Reference 1 — Data Integrity, Weather Decoding & Feature Engineering",
        subtitle="Exact identities and periodic encodings implemented in src/data_validation.py and src/features.py",
        rows=[
            (
                "1. Additivity & Non-Negativity Identity",
                r"$\mathrm{cnt}_{d,h} = \mathrm{casual}_{d,h} + \mathrm{registered}_{d,h}, \quad \mathrm{cnt}_{d,h} \geq 0 \quad (0\ \mathrm{violations\ across}\ 17,379\ \mathrm{hours})$",
                "Target leakage check: casual & registered excluded from X",
            ),
            (
                "2. Cyclical Fourier Encoding (Hour, Month, Weekday)",
                r"$\mathrm{hr\_sin} = \sin\left(\frac{2\pi \cdot \mathrm{hr}}{24}\right), \quad \mathrm{hr\_cos} = \cos\left(\frac{2\pi \cdot \mathrm{hr}}{24}\right), \quad \mathrm{mnth\_sin} = \sin\left(\frac{2\pi(\mathrm{mnth}-1)}{12}\right), \quad \mathrm{mnth\_cos} = \cos\left(\frac{2\pi(\mathrm{mnth}-1)}{12}\right)$",
                "Preserves periodic continuity between hour 23 and hour 0",
            ),
            (
                "3. Physical Weather De-Normalization (UCI Scales)",
                r"$T_{\mathrm{C}} = 41 \cdot \mathrm{temp}, \quad AT_{\mathrm{C}} = 50 \cdot \mathrm{atemp}, \quad H_{\%} = 100 \cdot \mathrm{hum}, \quad W_{\mathrm{km/h}} = 67 \cdot \mathrm{windspeed}$",
                "Converts [0, 1] normalized sensors to physical units",
            ),
            (
                "4. Rush-Hour Window & Weather Severity Index",
                r"$\mathcal{W}_{\mathrm{commute}} = \{7, 8\} \cup \{16, 17, 18\}, \quad \mathrm{Severity}_{d,h} = 0.5 \cdot \left(\frac{\mathrm{weathersit}_{d,h}-1}{3}\right) + 0.3 \cdot \mathrm{hum}_{d,h} + 0.2 \cdot \mathrm{windspeed}_{d,h}$",
                "Captures directional rush-hour peaks & adverse weather",
            ),
        ],
        figsize=(12.8, 5.2),
    )

    # Card 2: Statistical Hypothesis Testing Formulas
    eq2_path = FIG_DIR / "eq_hypothesis_testing.png"
    _draw_card(
        eq2_path,
        title="Mathematical Reference 2 — Hypothesis Formulation & Econometric Testing",
        subtitle="Daily commuting-window aggregation and 4-tier statistical verification implemented in src/hypothesis.py",
        rows=[
            (
                "5. Daily Commuting-Window Rental Total (n = 731 Days)",
                r"$Y_d = \sum_{h \in \{7, 8, 16, 17, 18\}} \mathrm{cnt}_{d,h}, \quad H_0: \mu_{\mathrm{work}} = \mu_{\mathrm{nonwork}} \quad \mathrm{vs.} \quad H_1: \mu_{\mathrm{work}} > \mu_{\mathrm{nonwork}} \quad (\alpha = 0.05)$",
                "Working days (n₁=500): mean 2,068.5 vs. Non-working (n₀=231): mean 1,105.2",
            ),
            (
                "6. Welch's Unequal-Variance t-Test & Satterthwaite df",
                r"$t = \frac{\bar{Y}_1 - \bar{Y}_0}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_0^2}{n_0}}} = 17.958, \quad \nu = \frac{(s_1^2/n_1 + s_0^2/n_0)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_0^2/n_0)^2}{n_0-1}} = 645.6, \quad p = 4.33 \times 10^{-59}$",
                "Difference Δ = +963.3 rides/day (95% CI: [858.0, 1,068.7])",
            ),
            (
                "7. Standardized Effect Size (Hedges' g & Rank-Biserial r)",
                r"$g = J(\nu) \cdot \frac{\bar{Y}_1 - \bar{Y}_0}{\sqrt{\frac{(n_1-1)s_1^2 + (n_0-1)s_0^2}{n_1+n_0-2}}} = 1.231 \ (\mathrm{Very\ Large}), \quad U = 94,178.5, \quad r_{\mathrm{rb}} = \frac{2U}{n_1 n_0} - 1 = 0.631$",
                "Confirmed non-parametrically via Mann-Whitney U (p = 3.65e-43)",
            ),
            (
                "8. Newey-West HAC Semilog Regression (7-Day Lag Bandwidth)",
                r"$\log(1 + Y_d) = \beta_0 + \beta_1 \mathrm{workingday}_d + \beta_2 \mathrm{temp}_d + \beta_3 \mathrm{temp}_d^2 + \beta_4 \mathrm{hum}_d + \beta_5 \mathrm{windspeed}_d + \gamma_{\mathrm{weather}} + \delta_{\mathrm{mnth}} + \theta \mathrm{yr}_d + \varepsilon_d$",
                "HAC β₁ = +0.6372 (z = 19.50, p = 5.85e-85) → exp(β₁) = 1.89x multiplier",
            ),
        ],
        figsize=(12.8, 5.2),
    )

    # Card 3: Demand Estimation Model, Metrics & Anomaly Detection Formulas
    eq3_path = FIG_DIR / "eq_model_and_metrics.png"
    _draw_card(
        eq3_path,
        title="Mathematical Reference 3 — Demand Estimation, Evaluation Metrics & Anomaly Detection",
        subtitle="Log-space gradient boosting, uncertainty intervals, and MAD outlier scoring in src/train_model.py & src/anomaly.py",
        rows=[
            (
                "9. Log1p Target Transformation & Non-Negative Inverse Prediction",
                r"$z_{d,h} = \log(1 + \mathrm{cnt}_{d,h}), \quad \hat{y}_{d,h} = \max\left(0,\ \exp(\hat{z}_{d,h}) - 1\right)$",
                "Stabilizes right-skewed variance & guarantees non-negative counts",
            ),
            (
                "10. Out-of-Sample Evaluation Metrics (Held-Out Test: Jul–Dec 2012)",
                r"$\mathrm{MAE} = \frac{1}{N}\sum_{i=1}^{N} |y_i - \hat{y}_i| = 43.54, \quad \mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2} = 68.18, \quad R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} = 0.9037$",
                "Reduces MAE by 59.4% and RMSE by 54.7% vs. Baseline",
            ),
            (
                "11. Empirical Prediction Interval Band (10th–90th Residual Quantiles)",
                r"$\left[\hat{L}_{d,h},\ \hat{U}_{d,h}\right] = \left[\max\left(0,\ \hat{y}_{d,h} + q_{0.10}(r_{\mathrm{bucket}})\right),\ \max\left(0,\ \hat{y}_{d,h} + q_{0.90}(r_{\mathrm{bucket}})\right)\right], \quad r_i = y_i - \hat{y}_i$",
                "Achieves 79.8% empirical coverage on held-out 2012 test data",
            ),
            (
                "12. Cross-Fitted Robust MAD Z-Score for Daily Anomaly Detection",
                r"$r_d = y_d - \hat{y}_d^{\mathrm{OOF}}, \quad \mathrm{MAD} = \mathrm{median}\left(|r_d - \mathrm{median}(r)|\right), \quad z_{\mathrm{robust},d} = \frac{0.6745 \cdot (r_d - \mathrm{median}(r))}{\mathrm{MAD}}, \quad |z_{\mathrm{robust},d}| \geq 3.0$",
                "Detects 12 genuine regime shocks (e.g., Hurricane Sandy 2012-10-29)",
            ),
            (
                "13. Station-Level Net Flow Conservation (Phase 2 Rebalancing Target)",
                r"$\Delta S_{s,t} = \mathrm{Inflow}_{s,t} - \mathrm{Outflow}_{s,t}, \quad I_{s,t+1} = \min\left(C_s,\ \max\left(0,\ I_{s,t} + \Delta S_{s,t} + R_{s,t}\right)\right)$",
                "Links aggregate fleet demand to truck rebalancing schedule R(s,t)",
            ),
        ],
        figsize=(12.8, 6.25),
    )

    arch_path = FIG_DIR / "system_architecture.png"
    draw_architecture_diagram(arch_path)

    for p in (eq1_path, eq2_path, eq3_path, arch_path):
        if ARTIFACT_DIR.exists():
            shutil.copy2(p, ARTIFACT_DIR / p.name)
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
