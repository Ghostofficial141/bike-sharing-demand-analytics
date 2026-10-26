"""Generate a 100% Black-and-White (monochrome), human-styled academic/engineering project report
in PDF, HTML, and Word (.docx / .doc) formats for:
    Name: AMAN KUMAR RAJ
    Roll No.: 23CEB0A27
"""
from __future__ import annotations

import base64
import io
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config

BW_FIG_DIR = config.FIGURES_DIR / "bw"
BW_FIG_DIR.mkdir(parents=True, exist_ok=True)


def to_grayscale_png_bytes(img_path: Path) -> bytes:
    """Convert any RGB/RGBA PNG chart into a pure grayscale (black & white) PNG."""
    arr = mpimg.imread(str(img_path))
    if arr.ndim == 3:
        rgb = arr[..., :3]
        # ITU-R BT.601 luminance weights
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        gray = arr

    # Slight contrast stretch so lines and text are crisp in black & white print
    gray = np.clip((gray - 0.5) * 1.08 + 0.5, 0.0, 1.0)

    bw_out_path = BW_FIG_DIR / img_path.name
    plt.imsave(str(bw_out_path), gray, cmap="gray", vmin=0.0, vmax=1.0, format="png")
    return bw_out_path.read_bytes()


def fig_tag(fname: str, fig_num: int, caption: str) -> str:
    """Return an HTML <figure> with pure black-and-white base64 image and academic caption."""
    img_path = config.FIGURES_DIR / fname
    if not img_path.exists():
        return ""
    bw_bytes = to_grayscale_png_bytes(img_path)
    b64 = base64.b64encode(bw_bytes).decode("ascii")
    return f"""
    <figure class="report-fig">
      <img src="data:image/png;base64,{b64}" alt="Figure {fig_num}"/>
      <figcaption><strong>Figure {fig_num}.</strong> {caption}</figcaption>
    </figure>
    """


def build_html() -> str:
    fig1 = fig_tag(
        "data_quality_gaps_and_daily_trend.png",
        1,
        "Daily total bike rentals across 2011–2012 (top) and distribution of the 165 missing hourly timestamps across 76 calendar dates (bottom).",
    )
    fig2 = fig_tag(
        "eda_hourly_and_seasonal.png",
        2,
        "Mean hourly rental profiles comparing working vs. non-working days (left) and across the four dataset season codes (right).",
    )
    fig3 = fig_tag(
        "eda_weather_and_temperature.png",
        3,
        "Mean hourly bike rentals across weather situation categories (left) and ambient temperature bands in degrees Celsius (right).",
    )
    fig4 = fig_tag(
        "hypothesis_commute.png",
        4,
        "Comparison of daily commuting-window rentals (07:00–08:59 and 16:00–18:59) between working days (n = 500) and non-working days (n = 231).",
    )
    fig5 = fig_tag(
        "model_actual_vs_predicted.png",
        5,
        "Observed versus predicted hourly bike rentals on the held-out test set (July–December 2012, n = 4,376 hours).",
    )
    fig6 = fig_tag(
        "model_residuals.png",
        6,
        "Residual diagnostics on the held-out test period: residuals vs. predicted values (left), residual histogram (center), and hourly MAE comparison against the baseline model (right).",
    )
    fig7 = fig_tag(
        "model_time_series.png",
        7,
        "Chronological comparison of actual rentals, gradient boosting predictions, and baseline estimates over a two-week test sample (top) and daily aggregated totals across July–December 2012 (bottom).",
    )
    fig8 = fig_tag(
        "model_permutation_importance.png",
        8,
        "Permutation feature importance measured by increase in out-of-sample test RMSE (July–December 2012).",
    )
    fig9 = fig_tag(
        "anomaly_detection.png",
        9,
        "Observed daily rentals versus cross-fitted conditional expectation (top) and robust MAD residual z-scores highlighting major weather and holiday anomalies (bottom).",
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Bike Sharing Demand Analysis & Operations Planning Report — Aman Kumar Raj (23CEB0A27)</title>
<style>
  @page {{
    size: A4;
    margin: 18mm 18mm 18mm 18mm;
  }}
  * {{
    box-sizing: border-box;
    color: #000000 !important;
    background-color: transparent;
  }}
  body {{
    font-family: "Times New Roman", Times, Georgia, serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #000000;
    background: #ffffff;
    max-width: 820px;
    margin: 24px auto;
    padding: 0 16px;
    text-align: justify;
  }}
  .title-block {{
    text-align: center;
    border-top: 2px solid #000000;
    border-bottom: 2px solid #000000;
    padding: 16px 8px;
    margin-bottom: 20px;
  }}
  .title-block h1 {{
    font-size: 17pt;
    font-weight: bold;
    margin: 0 0 10px 0;
    line-height: 1.25;
    text-align: center;
  }}
  .title-block .subtitle {{
    font-size: 11.5pt;
    font-style: italic;
    margin: 0 0 12px 0;
    text-align: center;
  }}
  .title-block .author-line {{
    font-size: 11.5pt;
    margin: 4px 0;
    text-align: center;
  }}
  h2 {{
    font-size: 13pt;
    font-weight: bold;
    margin-top: 22px;
    margin-bottom: 8px;
    border-bottom: 1px solid #000000;
    padding-bottom: 3px;
    text-align: left;
    page-break-after: avoid;
    break-after: avoid;
  }}
  h3 {{
    font-size: 11.5pt;
    font-weight: bold;
    margin-top: 15px;
    margin-bottom: 6px;
    text-align: left;
    page-break-after: avoid;
    break-after: avoid;
  }}
  p {{
    margin: 0 0 9px 0;
  }}
  ul, ol {{
    margin: 4px 0 10px 22px;
    padding: 0;
  }}
  li {{
    margin-bottom: 4px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 16px 0;
    font-size: 9.8pt;
    border-top: 1.5px solid #000000;
    border-bottom: 1.5px solid #000000;
    page-break-inside: auto;
  }}
  tr {{
    page-break-inside: avoid;
    break-inside: avoid;
  }}
  thead th {{
    border-bottom: 1px solid #000000;
    padding: 6px 7px;
    text-align: left;
    font-weight: bold;
    vertical-align: bottom;
  }}
  tbody td {{
    border-bottom: 0.5px solid #666666;
    padding: 5px 7px;
    text-align: left;
    vertical-align: top;
  }}
  tbody tr:last-child td {{
    border-bottom: none;
  }}
  .num {{
    text-align: right;
  }}
  .table-caption {{
    font-size: 9.8pt;
    margin: 10px 0 4px 0;
    text-align: left;
    page-break-after: avoid;
    break-after: avoid;
  }}
  .eq-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 10px 0 10px 24px;
    font-family: "Cambria Math", "Times New Roman", serif;
    font-size: 10.8pt;
    page-break-inside: avoid;
    break-inside: avoid;
  }}
  .eq-num {{
    font-style: normal;
    margin-left: 16px;
  }}
  .report-fig {{
    margin: 14px 0 16px 0;
    text-align: center;
    page-break-inside: avoid;
    break-inside: avoid;
  }}
  .report-fig img {{
    max-width: 98%;
    height: auto;
    border: 0.75px solid #000000;
    filter: grayscale(100%);
  }}
  .report-fig figcaption {{
    font-size: 9.5pt;
    margin-top: 5px;
    text-align: justify;
  }}
  code {{
    font-family: "Courier New", Courier, monospace;
    font-size: 9.6pt;
  }}
  pre {{
    font-family: "Courier New", Courier, monospace;
    font-size: 9pt;
    border: 0.75px solid #000000;
    padding: 8px 10px;
    margin: 8px 0 12px 0;
    white-space: pre-wrap;
    page-break-inside: avoid;
    break-inside: avoid;
  }}
  @media print {{
    body {{
      margin: 0;
      padding: 0;
      max-width: 100%;
    }}
    .report-fig img {{
      max-height: 195mm;
    }}
  }}
</style>
</head>
<body>

<div class="title-block">
  <h1>Investigation of Bike-Sharing Demand Patterns, Fleet Allocation, and Hourly Demand Estimation</h1>
  <div class="subtitle">Empirical Analysis and Decision-Support Prototype on the Capital Bikeshare Dataset (2011–2012)</div>
  <div class="author-line"><strong>Submitted by:</strong> AMAN KUMAR RAJ &nbsp;&nbsp;|&nbsp;&nbsp; <strong>Roll No.:</strong> 23CEB0A27</div>
  <div class="author-line"><strong>Dataset:</strong> UCI Machine Learning Repository — Bike Sharing Dataset (17,379 Hourly &amp; 731 Daily Observations)</div>
</div>

<h2>1. Introduction and Problem Statement</h2>
<p>
Urban bike-sharing networks frequently encounter an operational imbalance in which riders report empty docks at high-demand stations while a substantial number of bicycles remain unutilized elsewhere in the city. The objective of this study is to examine two years of historical operational data from the Capital Bikeshare system in Washington, D.C. (2011–2012) and determine whether addressing rider complaints primarily requires acquiring additional bicycles or improving fleet reallocation and shift planning.
</p>
<p>
To answer this question rigorously, this work is organized around four core tasks: (1) systematic data quality auditing and leakage-free feature engineering on the raw hourly and daily logs; (2) pre-registered statistical hypothesis testing on commuting-window demand; (3) development and chronological validation of an hourly demand-estimation model; and (4) implementation of an interactive Streamlit prototype equipped with a natural-language query engine that allows operations managers to inspect demand profiles and test scheduling scenarios.
</p>

<h3>1.1 Summary of Principal Findings</h3>
<p>
Across the 731-day study period (<code>3,292,679</code> total checkouts over <code>17,379</code> recorded hours), the overall mean demand is only <code>189.5</code> rentals per hour (median <code>142.0</code> rentals/hour), whereas the 99th-percentile hourly demand reaches <code>782</code> to <code>977</code> rentals per hour. On working days (<code>n = 500</code> days), <strong>51.3% of all daily checkouts occur within just five commuting hours</strong> (<code>07:00–08:59</code> and <code>16:00–18:59</code>), peaking at <code>477.0</code> rentals/hour at <code>08:00</code> (90th percentile <code>692</code>) and <code>525.3</code> rentals/hour at <code>17:00</code> (90th percentile <code>782</code>). Between these peaks, demand falls by more than 60% during midday (<code>10:00–15:00</code>) and by more than 95% overnight (<code>00:00–05:00</code>). On weekends and holidays (<code>n = 231</code> days), the morning commuter spike disappears (<code>105.7</code> rentals/hour at <code>08:00</code>) and ridership shifts to a single midday recreational plateau between <code>12:00</code> and <code>16:00</code> averaging <code>359–373</code> rentals/hour.
</p>
<p>
Consequently, <strong>simultaneous rider complaints and idle bicycles are primarily a temporal reallocation problem rather than a 24-hour fleet deficit</strong>. Purchasing additional bicycles without modifying truck-rebalancing schedules would leave more inventory idle during the 19 off-peak hours of each working day. At the same time, total annual ridership increased by <strong>64.9%</strong> from <code>1,243,103</code> rentals in 2011 to <code>2,049,576</code> rentals in 2012 (with registered subscribers generating <code>81.2%</code> of rides). Thus, while daily stockouts should be addressed immediately through pre-peak staging before <code>07:00</code> and inter-peak rebalancing between <code>10:00</code> and <code>15:30</code>, medium-term capacity expansion at major commuter corridors is warranted to keep pace with secular system growth.
</p>

<h2>2. Audit of Prior Time-Series Report and Methodological Corrections</h2>
<p>
Prior to building the Python pipeline, I audited the existing R Markdown report (<code>time-series-analysis.html</code>) included in the workspace. Although that reference report provided a useful exploratory overview of daily rentals, it contained five methodological limitations that prevented its direct use for operational forecasting:
</p>
<ol>
  <li><strong>Preprocessing Data Leakage Across Train and Test Sets:</strong> The legacy script applied outlier cleaning (<code>forecast::tsclean()</code>), centered moving averages (<code>ma(order = 7)</code> and <code>ma(order = 30)</code>), and seasonal-trend decomposition (<code>stl()</code>) to the entire 731-day series prior to slicing the holdout window (indices <code>700:725</code>). Centered windows at day <i>t</i> incorporate future observations <i>t + 1, ..., t + k</i>, leaking holdout information into the training series.</li>
  <li><strong>Omission of Hourly Granularity:</strong> The legacy report analyzed only <code>day.csv</code> (731 rows) and ignored <code>hour.csv</code> (17,379 rows). Daily aggregation conceals the morning and evening rush-hour spikes that drive dock stockouts.</li>
  <li><strong>Removal of Genuine Extreme Events:</strong> Applying <code>tsclean()</code> prior to evaluation replaced real operational shocks—such as the Hurricane Sandy shutdown on 29 October 2012—with smoothed interpolations, distorting holdout error metrics.</li>
  <li><strong>Misspecified Seasonal Period:</strong> Daily series were defined using <code>ts(..., frequency = 30)</code>, imposing an artificial 30-day cycle rather than the true 7-day weekly and 365.25-day annual periodicities.</li>
  <li><strong>Absence of Weather and Calendar Predictors:</strong> The univariate <code>auto.arima(1,1,1)</code> specification ignored temperature, precipitation, humidity, wind speed, and working-day indicators.</li>
</ol>
<p>
In the present implementation, the original HTML file is preserved intact for reference, and a clean modular Python pipeline (<code>src/</code>) is constructed from scratch with strict chronological splitting and zero target leakage.
</p>

<h2>3. Data Quality Verification and Feature Engineering</h2>

<h3>3.1 Automated Data Quality Checks</h3>
<p>
The raw UCI files (<code>data/raw/hour.csv</code> and <code>data/raw/day.csv</code>) were validated using 26 programmatic checks in <code>src/data_validation.py</code>. Table 1 summarizes the principal audit findings.
</p>

<div class="table-caption"><strong>Table 1.</strong> Summary of automated data quality checks performed on <code>hour.csv</code> and <code>day.csv</code>.</div>
<table>
  <thead>
    <tr>
      <th>Check Category</th>
      <th>Verification Rule</th>
      <th>Observed Result</th>
      <th>Analytical Treatment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Dimensions &amp; Schema</td>
      <td>Match UCI specification (17 hourly cols, 16 daily cols)</td>
      <td><code>hour.csv</code>: 17,379 &times; 17<br/><code>day.csv</code>: 731 &times; 16</td>
      <td>Exact match verified; raw files kept read-only.</td>
    </tr>
    <tr>
      <td>Missing &amp; Duplicate Rows</td>
      <td>Check NaN cells, duplicate rows, duplicate <code>instant</code>, and duplicate <code>(dteday, hr)</code></td>
      <td>0 missing values, 0 duplicates, strictly monotonic timestamps</td>
      <td>No records dropped or altered.</td>
    </tr>
    <tr>
      <td>Component Additivity</td>
      <td>Verify <code>casual + registered == cnt</code> and non-negativity (<code>cnt &ge; 0</code>)</td>
      <td>0 mismatches across all 17,379 hourly and 731 daily records</td>
      <td><code>casual</code> and <code>registered</code> excluded from predictors to prevent target leakage.</td>
    </tr>
    <tr>
      <td>Cross-File Consistency</td>
      <td>Aggregate <code>hour.csv</code> by <code>dteday</code> and compare sums against <code>day.csv</code></td>
      <td>0 dates disagree (maximum absolute deviation = 0 rentals)</td>
      <td>Confirmed <code>day.csv</code> is the exact daily sum of <code>hour.csv</code>.</td>
    </tr>
    <tr>
      <td>Hourly Sequence Gaps</td>
      <td>Compare 17,379 rows against full <code>731 &times; 24 = 17,544</code> calendar grid</td>
      <td>165 absent hours across 76 dates (655 complete 24-hr days; 62 with 23 hrs; 14 with &le;22 hrs)</td>
      <td>Documented, not imputed with synthetic zeros; largest gap is 23 missing hours on 2012-10-29 (Hurricane Sandy).</td>
    </tr>
    <tr>
      <td>Season Code Audit</td>
      <td>Inspect calendar dates of <code>season</code> transitions (codes 1..4)</td>
      <td>Transitions occur on equinox/solstice dates (~Mar 21, Jun 21, Sep 23, Dec 21)</td>
      <td>Code 1 (labeled <code>springer</code> in README) corresponds to Dec 21–Mar 20; documented in UI and reports.</td>
    </tr>
    <tr>
      <td>Weather Sensor Audit</td>
      <td>Check physical bounds on <code>temp, atemp, hum, windspeed &isin; [0, 1]</code></td>
      <td>22 hours on <code>2011-03-10</code> record <code>hum = 0.0</code> during light rain; 24 hours on <code>2012-08-17</code> show constant <code>atemp = 0.2424</code></td>
      <td>Flagged as sensor anomalies in quality report; raw values retained for reproducibility.</td>
    </tr>
    <tr>
      <td>Distributional Outliers</td>
      <td>Global IQR rule (<code>cnt &gt; 642.5</code>) and conditional robust z-score (<code>|z_robust| &gt; 3.5</code>)</td>
      <td>505 global IQR outliers; 438 conditional outliers within <code>hr &times; workingday</code></td>
      <td>Retained in dataset because high values represent genuine rush-hour demand peaks.</td>
    </tr>
  </tbody>
</table>

{fig1}

<h3>3.2 Feature Engineering and Leakage Prevention</h3>
<p>
All feature transformations are implemented in <code>src/features.py</code> as deterministic row-wise functions of calendar and weather variables so that no future or target information leaks across observations. First, the normalized weather measurements are rescaled to physical units for interpretability:
</p>
<div class="eq-row">
  <span><i>T</i><sub>C</sub> = 41 &middot; <code>temp</code>, &nbsp;&nbsp; <i>AT</i><sub>C</sub> = 50 &middot; <code>atemp</code>, &nbsp;&nbsp; <i>H</i><sub>%</sub> = 100 &middot; <code>hum</code>, &nbsp;&nbsp; <i>W</i><sub>km/h</sub> = 67 &middot; <code>windspeed</code></span>
  <span class="eq-num">(1)</span>
</div>
<p>
Second, to preserve periodic continuity across midnight (hour 23 to hour 0), week boundaries, and year-end transitions, cyclical Fourier encodings are constructed for hour of day, day of week, and month:
</p>
<div class="eq-row">
  <span><code>hr_sin</code> = sin(2&pi; &middot; <code>hr</code> / 24), &nbsp;&nbsp; <code>hr_cos</code> = cos(2&pi; &middot; <code>hr</code> / 24)</span>
  <span class="eq-num">(2)</span>
</div>
<div class="eq-row">
  <span><code>weekday_sin</code> = sin(2&pi; &middot; <code>weekday</code> / 7), &nbsp;&nbsp; <code>weekday_cos</code> = cos(2&pi; &middot; <code>weekday</code> / 7)</span>
  <span class="eq-num">(3)</span>
</div>
<div class="eq-row">
  <span><code>mnth_sin</code> = sin(2&pi; &middot; (<code>mnth</code> &minus; 1) / 12), &nbsp;&nbsp; <code>mnth_cos</code> = cos(2&pi; &middot; (<code>mnth</code> &minus; 1) / 12)</span>
  <span class="eq-num">(4)</span>
</div>
<p>
Third, domain-specific indicators are defined for weekends (<code>is_weekend = 1</code> if <code>weekday &isin; {{0, 6}}</code>), commuting windows (<code>is_commute_hour = 1</code> if <code>hr &isin; {{7, 8, 16, 17, 18}}</code>), and their interaction with working-day status (<code>workday_commute = is_commute_hour &times; workingday</code>). Because <code>cnt = casual + registered</code> holds identically for every record, including either component count as a predictor would introduce target leakage. The helper function <code>assert_no_leakage()</code> enforces at runtime that none of <code>(casual, registered, cnt, instant)</code> can be included in the model feature matrix.
</p>

{fig2}
{fig3}

<h2>4. Hypothesis Formulation and Statistical Testing</h2>

<h3>4.1 Experimental Design and Hypotheses</h3>
<p>
To test whether commuting-window demand is structurally higher on working days than on weekends and holidays, the commuting windows were fixed prior to testing as Morning (<code>07:00–08:59</code>, <code>hr &isin; {{7, 8}}</code>) and Evening (<code>16:00–18:59</code>, <code>hr &isin; {{16, 17, 18}}</code>). Because hourly observations within the same day are correlated, the unit of analysis is defined as a single calendar date <i>d</i> (<i>N</i> = 731 days), with daily commuting-window volume given by:
</p>
<div class="eq-row">
  <span><i>Y<sub>d</sub></i> = &Sigma;<sub><i>h</i> &isin; {{7, 8, 16, 17, 18}}</sub> <code>cnt</code><sub><i>d,h</i></sub></span>
  <span class="eq-num">(5)</span>
</div>
<p>
Let &mu;<sub>1</sub> and &mu;<sub>0</sub> denote the population mean commuting-window rental totals on working days (<code>workingday = 1</code>, <i>n</i><sub>1</sub> = 500) and non-working days (<code>workingday = 0</code>, <i>n</i><sub>0</sub> = 231), respectively. We test at significance level &alpha; = 0.05:
</p>
<ul>
  <li><strong>Null Hypothesis (<i>H</i><sub>0</sub>):</strong> &mu;<sub>1</sub> = &mu;<sub>0</sub> (mean daily commuting-window demand is equal on working and non-working days).</li>
  <li><strong>Alternative Hypothesis (<i>H</i><sub>1</sub>, one-sided):</strong> &mu;<sub>1</sub> &gt; &mu;<sub>0</sub> (mean daily commuting-window demand is higher on working days).</li>
</ul>
<p>
Because Levene's test indicates unequal group variances (<i>p</i> = 6.34 &times; 10<sup>&minus;10</sup>; SD<sub>1</sub> = 862.9 vs. SD<sub>0</sub> = 566.4), the primary parametric test is Welch's unequal-variance <i>t</i>-test with Satterthwaite degrees of freedom &nu;, supplemented by the standardized effect size (Hedges' <i>g</i>) and the nonparametric Mann-Whitney <i>U</i> test:
</p>
<div class="eq-row">
  <span><i>t</i> = (<i>Y&#772;</i><sub>1</sub> &minus; <i>Y&#772;</i><sub>0</sub>) / &radic;(<i>s</i><sub>1</sub><sup>2</sup>/<i>n</i><sub>1</sub> + <i>s</i><sub>0</sub><sup>2</sup>/<i>n</i><sub>0</sub>), &nbsp;&nbsp;&nbsp;&nbsp; <i>g</i> = <i>J</i>(&nu;) &middot; (<i>Y&#772;</i><sub>1</sub> &minus; <i>Y&#772;</i><sub>0</sub>) / <i>s</i><sub>pooled</sub></span>
  <span class="eq-num">(6)</span>
</div>
<p>
Furthermore, because daily residuals exhibit positive serial correlation across consecutive days (lag-1 autocorrelation <i>r</i><sub>1</sub> = 0.79, Durbin-Watson = 0.41) and weather varies across the year, I also estimate a semi-logarithmic Ordinary Least Squares (OLS) regression with Newey-West heteroskedasticity- and autocorrelation-consistent (HAC) standard errors (7-day lag bandwidth):
</p>
<div class="eq-row">
  <span>log(1 + <i>Y<sub>d</sub></i>) = &beta;<sub>0</sub> + &beta;<sub>1</sub><code>workingday</code><sub><i>d</i></sub> + &beta;<sub>2</sub><code>temp</code><sub><i>d</i></sub> + &beta;<sub>3</sub><code>temp</code><sub><i>d</i></sub><sup>2</sup> + &beta;<sub>4</sub><code>hum</code><sub><i>d</i></sub> + &beta;<sub>5</sub><code>windspeed</code><sub><i>d</i></sub> + &gamma;<sub>weather</sub> + &delta;<sub>month</sub> + &theta;<code>yr</code><sub><i>d</i></sub> + &varepsilon;<sub><i>d</i></sub></span>
  <span class="eq-num">(7)</span>
</div>

<h3>4.2 Empirical Results</h3>
<p>
Table 2 reports the test statistics across the full 731-day sample and four sensitivity subsets. On working days, commuting-window rentals average <code>2,068.5</code> rides/day (median <code>2,078.0</code>) compared to <code>1,105.2</code> rides/day (median <code>1,096.0</code>) on non-working days—an increase of <code>+963.3</code> rides/day (95% Welch CI: <code>[858.0, 1,068.7]</code>; 10,000-draw bootstrap 95% CI: <code>[857.3, 1,069.2]</code>). Welch's test yields <i>t</i> = 17.96 (df = 645.6, one-sided <i>p</i> = 4.33 &times; 10<sup>&minus;59</sup>) with Hedges' <i>g</i> = 1.23 (very large effect size). The Mann-Whitney test confirms this finding (<i>U</i> = 94,178.5, one-sided <i>p</i> = 3.65 &times; 10<sup>&minus;43</sup>, rank-biserial <i>r</i> = 0.631).
</p>

<div class="table-caption"><strong>Table 2.</strong> Commuting-window hypothesis test results across primary and sensitivity samples.</div>
<table>
  <thead>
    <tr>
      <th>Sample / Subset</th>
      <th class="num"><i>n</i><sub>work</sub> / <i>n</i><sub>non-work</sub></th>
      <th class="num">Working Mean (Median)</th>
      <th class="num">Non-Working Mean (Median)</th>
      <th class="num">Mean Diff. (95% CI)</th>
      <th class="num">Welch <i>t</i> (df)</th>
      <th class="num">One-Sided <i>p</i></th>
      <th class="num">Hedges' <i>g</i></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Primary (All 731 Days)</td>
      <td class="num">500 / 231</td>
      <td class="num">2,068.5 (2,078.0)</td>
      <td class="num">1,105.2 (1,096.0)</td>
      <td class="num">+963.3 (858.0, 1,068.7)</td>
      <td class="num">17.96 (645.6)</td>
      <td class="num">4.33 &times; 10<sup>&minus;59</sup></td>
      <td class="num">1.23</td>
    </tr>
    <tr>
      <td>Complete 5-Hr Windows Only</td>
      <td class="num">495 / 230</td>
      <td class="num">2,087.3 (2,093.0)</td>
      <td class="num">1,109.4 (1,100.0)</td>
      <td class="num">+977.9 (873.4, 1,082.4)</td>
      <td class="num">18.38 (636.2)</td>
      <td class="num">3.89 &times; 10<sup>&minus;61</sup></td>
      <td class="num">1.27</td>
    </tr>
    <tr>
      <td>Year 2011 Only (<code>yr = 0</code>)</td>
      <td class="num">250 / 115</td>
      <td class="num">1,535.6 (1,739.0)</td>
      <td class="num">858.6 (880.0)</td>
      <td class="num">+677.0 (570.6, 783.4)</td>
      <td class="num">12.52 (302.6)</td>
      <td class="num">1.48 &times; 10<sup>&minus;29</sup></td>
      <td class="num">1.24</td>
    </tr>
    <tr>
      <td>Year 2012 Only (<code>yr = 1</code>)</td>
      <td class="num">250 / 116</td>
      <td class="num">2,601.4 (2,702.0)</td>
      <td class="num">1,349.6 (1,471.0)</td>
      <td class="num">+1,251.8 (1,108.8, 1,394.8)</td>
      <td class="num">17.23 (282.1)</td>
      <td class="num">2.87 &times; 10<sup>&minus;46</sup></td>
      <td class="num">1.77</td>
    </tr>
    <tr>
      <td>Weekends Only (Excl. Holidays)</td>
      <td class="num">500 / 210</td>
      <td class="num">2,068.5 (2,078.0)</td>
      <td class="num">1,107.0 (1,100.0)</td>
      <td class="num">+961.5 (855.2, 1,067.8)</td>
      <td class="num">17.76 (597.1)</td>
      <td class="num">2.72 &times; 10<sup>&minus;57</sup></td>
      <td class="num">1.23</td>
    </tr>
  </tbody>
</table>

<p>
In the Newey-West HAC regression (Equation 7, <i>R</i><sup>2</sup> = 0.682), the working-day coefficient is &beta;&#770;<sub>1</sub> = +0.6372 (HAC <i>z</i> = 19.50, one-sided <i>p</i> = 5.85 &times; 10<sup>&minus;85</sup>), indicating that working days generate exp(0.6372) = <strong>1.89&times;</strong> (95% CI: <code>[1.77&times;, 2.02&times;]</code>) the commuting-window demand of non-working days after holding weather, month, and year constant. A secondary test comparing rainy/snowy days (at least one hour with <code>weathersit &ge; 3</code>, <i>n</i> = 300, mean <code>4,014.2</code> rentals/day) against dry/mist days (<i>n</i> = 431, mean <code>4,845.5</code> rentals/day) confirms that precipitation significantly depresses daily demand (Welch <i>t</i> = 5.89, <i>p</i> = 3.02 &times; 10<sup>&minus;9</sup>, stratified reduction = <code>&minus;828.7</code> rentals/day).
</p>

{fig4}

<h2>5. Hourly Demand-Estimation Model</h2>

<h3>5.1 Chronological Validation Strategy and Model Formulation</h3>
<p>
Because hourly bike demand is chronologically ordered and exhibits strong autocorrelation, random train/test splitting would leak contemporaneous weather and demand conditions across adjacent hours. I therefore partitioned <code>hour.csv</code> chronologically into an 18-month development set (<code>2011-01-01</code> through <code>2012-06-30</code>, <code>13,003</code> hours) and a 6-month held-out test set (<code>2012-07-01</code> through <code>2012-12-31</code>, <code>4,376</code> hours). Within the development period, hyperparameters were tuned using 4-fold expanding-window cross-validation (validating sequentially on Q3 2011, Q4 2011, Q1 2012, and Q2 2012).
</p>
<p>
Including the first six months of 2012 in the development window is necessary because total ridership expanded by 64.9% between 2011 (<code>yr = 0</code>) and 2012 (<code>yr = 1</code>). A diagnostic experiment trained solely on 2011 data confirmed that when <code>yr = 0</code> is constant during training, tree-based models cannot learn the 2012 level shift and incur a systematic mean bias of <code>&minus;89.65</code> rentals/hour on 2012 data (even though gradient boosting still improves <i>R</i><sup>2</sup> from <code>0.435</code> to <code>0.623</code> over the 2011 baseline).
</p>
<p>
Because hourly rental counts are right-skewed and non-negative, the primary model (<code>hgb_log</code>) fits a histogram-based gradient boosting regressor on log-transformed counts and applies a non-negative inverse transformation:
</p>
<div class="eq-row">
  <span><i>z<sub>d,h</sub></i> = log(1 + <code>cnt</code><sub><i>d,h</i></sub>), &nbsp;&nbsp;&nbsp;&nbsp; <i>y&#770;<sub>d,h</sub></i> = max(0, &nbsp;exp(<i>z&#770;<sub>d,h</sub></i>) &minus; 1)</span>
  <span class="eq-num">(8)</span>
</div>
<p>
Model accuracy on the held-out test set is evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination (<i>R</i><sup>2</sup>):
</p>
<div class="eq-row">
  <span>MAE = (1 / <i>N</i>) &Sigma;<sub><i>i</i>=1..<i>N</i></sub> |<i>y<sub>i</sub></i> &minus; <i>y&#770;<sub>i</sub></i>|, &nbsp;&nbsp;&nbsp;&nbsp; RMSE = &radic;[ (1 / <i>N</i>) &Sigma;<sub><i>i</i>=1..<i>N</i></sub> (<i>y<sub>i</sub></i> &minus; <i>y&#770;<sub>i</sub></i>)<sup>2</sup> ], &nbsp;&nbsp;&nbsp;&nbsp; <i>R</i><sup>2</sup> = 1 &minus; [ &Sigma;(<i>y<sub>i</sub></i> &minus; <i>y&#770;<sub>i</sub></i>)<sup>2</sup> / &Sigma;(<i>y<sub>i</sub></i> &minus; <i>y&#772;</i>)<sup>2</sup> ]</span>
  <span class="eq-num">(9)</span>
</div>
<p>
To quantify forecast uncertainty, out-of-fold development residuals (<i>r<sub>i</sub></i> = <i>y<sub>i</sub></i> &minus; <i>y&#770;<sub>i</sub></i>) are grouped into five predicted-demand quintiles, and the 10th and 90th empirical percentiles (<i>q</i><sub>0.10</sub>, <i>q</i><sub>0.90</sub>) are added to point predictions to form a non-negative 80% prediction interval:
</p>
<div class="eq-row">
  <span>[<i>L&#770;<sub>d,h</sub></i>, &nbsp;<i>U&#770;<sub>d,h</sub></i>] = [ max(0, &nbsp;<i>y&#770;<sub>d,h</sub></i> + <i>q</i><sub>0.10</sub>(<i>r</i><sub>bucket</sub>)), &nbsp;&nbsp;max(0, &nbsp;<i>y&#770;<sub>d,h</sub></i> + <i>q</i><sub>0.90</sub>(<i>r</i><sub>bucket</sub>)) ]</span>
  <span class="eq-num">(10)</span>
</div>

<h3>5.2 Model Evaluation Results</h3>
<p>
Table 3 compares the historical conditional baseline (mean rentals by <code>hr &times; workingday</code>), a log-target Random Forest (<code>rf_log</code>), a Poisson-loss HistGradientBoosting model (<code>hgb_poisson</code>), and the selected log-target HistGradientBoosting model (<code>hgb_log</code>). On the held-out July–December 2012 test period (4,376 hours, mean actual demand = <code>254.1</code> rentals/hour), <code>hgb_log</code> achieves <strong>MAE = 43.54 rentals/hour</strong>, <strong>RMSE = 68.18 rentals/hour</strong>, and <strong><i>R</i><sup>2</sup> = 0.904</strong>, reducing test MAE by <strong>59.4%</strong> and RMSE by <strong>54.7%</strong> relative to the baseline model. The empirical 80% prediction intervals cover <strong>79.8%</strong> of actual held-out test observations.
</p>

<div class="table-caption"><strong>Table 3.</strong> Expanding-window cross-validation and held-out test performance across candidate models.</div>
<table>
  <thead>
    <tr>
      <th>Model Specification</th>
      <th>Selected Hyperparameters (Dev CV)</th>
      <th class="num">Dev CV MAE</th>
      <th class="num">Dev CV RMSE</th>
      <th class="num">Test MAE</th>
      <th class="num">Test RMSE</th>
      <th class="num">Test <i>R</i><sup>2</sup></th>
      <th class="num">Test Bias</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Baseline (<code>hr &times; workingday</code> mean)</td>
      <td>Conditioned historical lookup table</td>
      <td class="num">73.46</td>
      <td class="num">105.22</td>
      <td class="num">107.21</td>
      <td class="num">150.35</td>
      <td class="num">0.532</td>
      <td class="num">&minus;87.26</td>
    </tr>
    <tr>
      <td>Random Forest (<code>rf_log</code>)</td>
      <td><code>n_estimators=200, max_features=1.0, min_samples_leaf=1</code></td>
      <td class="num">52.80</td>
      <td class="num">77.96</td>
      <td class="num">—</td>
      <td class="num">—</td>
      <td class="num">—</td>
      <td class="num">—</td>
    </tr>
    <tr>
      <td>HistGradientBoosting (<code>hgb_poisson</code>)</td>
      <td><code>loss=poisson, lr=0.05, max_leaf_nodes=31, min_leaf=50</code></td>
      <td class="num">46.09</td>
      <td class="num">69.64</td>
      <td class="num">—</td>
      <td class="num">—</td>
      <td class="num">—</td>
      <td class="num">—</td>
    </tr>
    <tr>
      <td><strong>Selected: <code>hgb_log</code> (Log-Target HGB)</strong></td>
      <td><code>lr=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2=1.0</code></td>
      <td class="num"><strong>45.78</strong></td>
      <td class="num"><strong>68.62</strong></td>
      <td class="num"><strong>43.54</strong></td>
      <td class="num"><strong>68.18</strong></td>
      <td class="num"><strong>0.904</strong></td>
      <td class="num"><strong>&minus;26.11</strong></td>
    </tr>
  </tbody>
</table>

{fig5}
{fig6}
{fig7}
{fig8}

<h2>6. Natural-Language Operations Prototype and Anomaly Detection</h2>

<h3>6.1 Interactive Prototype Architecture</h3>
<p>
To make the analytical results accessible to operations managers without programming, I developed a seven-tab Streamlit web application (<code>app.py</code>) paired with a deterministic natural-language query engine (<code>src/query_engine.py</code>). Rather than relying on an ungrounded generative model that could fabricate statistics, the query engine parses user questions in two stages: (1) an entity extractor (<code>extract_entities()</code>) identifies hours, time windows, weather conditions, temperature/humidity/wind values, day types, seasons, and years; and (2) an intent classifier (<code>detect_intent()</code>) routes the parsed parameters to 12 deterministic analytical handlers backed by <code>pandas</code> and the persisted <code>demand_model.joblib</code> pipeline. In addition, the simulator separates statistical point predictions from operational buffer policy by providing a configurable <strong>Operational Safety Buffer</strong> (default <code>+15%</code>). Table 4 illustrates representative queries and the exact responses computed by the engine.
</p>

<div class="table-caption"><strong>Table 4.</strong> Representative natural-language operations queries and deterministic outputs computed by <code>src/query_engine.py</code>.</div>
<table>
  <thead>
    <tr>
      <th>Manager Query</th>
      <th>Mapped Intent</th>
      <th>Computed Empirical Output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>"When is bike demand highest on working days?"</td>
      <td><code>peak_hours</code></td>
      <td>Peaks at <code>17:00</code> (mean <code>525</code> rentals/hr, p90 = <code>782</code>), followed by <code>18:00</code> (<code>492</code>/hr, p90 = <code>731</code>) and <code>08:00</code> (<code>477</code>/hr, p90 = <code>692</code>).</td>
    </tr>
    <tr>
      <td>"Compare demand on working days and non-working days."</td>
      <td><code>compare_day_types</code></td>
      <td>Working days average <code>4,585</code> rentals/day (bimodal commuter peaks, 84% registered) vs. <code>4,330</code> rentals/day on non-working days (midday plateau <code>12:00–16:00</code> at <code>359–373</code>/hr, 69% registered).</td>
    </tr>
    <tr>
      <td>"How does weather affect rental demand?"</td>
      <td><code>weather_effect</code></td>
      <td>Within <code>(hr, workingday, yr, mnth)</code> cells: Clear is <code>+8%</code> (<code>205</code>/hr), Mist is <code>&minus;7%</code> (<code>175</code>/hr), Light rain/snow is <code>&minus;40%</code> (<code>112</code>/hr), and Heavy rain/snow is <code>&minus;66%</code> (<code>74</code>/hr).</td>
    </tr>
    <tr>
      <td>"Estimate bike rentals at 8 AM in summer on a working day."</td>
      <td><code>predict</code></td>
      <td>Point forecast: <code>578</code> rentals/hr (80% interval: <code>461–691</code>; baseline: <code>477</code>). With <code>+15%</code> safety buffer: <code>665</code> bikes target.</td>
    </tr>
    <tr>
      <td>"Compare expected demand during clear weather and light rain."</td>
      <td><code>predict</code></td>
      <td>Working day in summer: Clear weather yields <code>6,003</code> rentals/day (peak <code>673</code>/hr at <code>17:00</code>) vs. Light rain <code>3,764</code> rentals/day (peak <code>451</code>/hr) — a <code>37%</code> reduction.</td>
    </tr>
    <tr>
      <td>"Which hours should the operations team monitor most closely?"</td>
      <td><code>monitor_hours</code></td>
      <td>Ranked by 90th-percentile load: (1) Working day <code>17:00</code> (<code>782</code>), (2) Working day <code>18:00</code> (<code>731</code>), (3) Working day <code>08:00</code> (<code>692</code>), and (4) Non-working day <code>13:00–15:00</code> (<code>590–605</code>).</td>
    </tr>
  </tbody>
</table>

<h3>6.2 Conditional Anomaly and Event Detection</h3>
<p>
As an extension to the core deliverables, <code>src/anomaly.py</code> identifies unusual system-wide demand shocks in <code>day.csv</code> using 8-fold contiguous block cross-fitting. For each day <i>d</i>, out-of-fold expected demand <i>y&#770;<sub>d</sub></i><sup>OOF</sup> is estimated given calendar and weather predictors, and residuals <i>r<sub>d</sub></i> = <i>y<sub>d</sub></i> &minus; <i>y&#770;<sub>d</sub></i><sup>OOF</sup> are standardized via the Median Absolute Deviation (MAD):
</p>
<div class="eq-row">
  <span>MAD = median(|<i>r<sub>d</sub></i> &minus; median(<i>r</i>)|), &nbsp;&nbsp;&nbsp;&nbsp; <i>z</i><sub>robust, <i>d</i></sub> = 0.6745 &middot; (<i>r<sub>d</sub></i> &minus; median(<i>r</i>)) / MAD</span>
  <span class="eq-num">(11)</span>
</div>
<p>
Flagging dates with |<i>z</i><sub>robust, <i>d</i></sub>| &ge; 3.0 or percentage shortfall &le; &minus;60% isolates 12 major operational anomalies (Table 5), including the Hurricane Sandy municipal shutdown on 29–30 October 2012 (<code>&minus;99.1%</code> and <code>&minus;61.6%</code>), the January 2011 winter blizzard (<code>&minus;82.2%</code>), unmodeled Christmas Eve demand drops (<code>2012-12-24</code>, coded as a regular working day in the UCI dataset but exhibiting a <code>&minus;70.5%</code> shortfall), and the 2012 National Cherry Blossom Festival / St. Patrick's Day surge (<code>+258.2%</code>).
</p>

<div class="table-caption"><strong>Table 5.</strong> Top positive and negative daily demand anomalies identified via cross-fitted residual screening.</div>
<table>
  <thead>
    <tr>
      <th>Date</th>
      <th class="num">Observed</th>
      <th class="num">Expected</th>
      <th class="num">Residual (% Diff.)</th>
      <th class="num">Robust <i>z</i></th>
      <th>Associated Historical Event in Washington, D.C.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>2012-10-29</code></td>
      <td class="num">22</td>
      <td class="num">2,451</td>
      <td class="num">&minus;2,429 (&minus;99.1%)</td>
      <td class="num">&minus;3.09</td>
      <td>Hurricane Sandy landfall (system suspended after 00:00; 23 missing hours)</td>
    </tr>
    <tr>
      <td><code>2011-01-27</code></td>
      <td class="num">431</td>
      <td class="num">2,418</td>
      <td class="num">&minus;1,987 (&minus;82.2%)</td>
      <td class="num">&minus;2.53</td>
      <td>January 2011 Mid-Atlantic winter snowstorm and icy road conditions</td>
    </tr>
    <tr>
      <td><code>2012-12-24</code></td>
      <td class="num">920</td>
      <td class="num">3,115</td>
      <td class="num">&minus;2,195 (&minus;70.5%)</td>
      <td class="num">&minus;2.80</td>
      <td>Christmas Eve (coded as <code>workingday = 1</code>, but commuters took holiday leave)</td>
    </tr>
    <tr>
      <td><code>2012-10-30</code></td>
      <td class="num">1,096</td>
      <td class="num">2,855</td>
      <td class="num">&minus;1,759 (&minus;61.6%)</td>
      <td class="num">&minus;2.24</td>
      <td>Hurricane Sandy Day 2 recovery and partial afternoon service resumption</td>
    </tr>
    <tr>
      <td><code>2012-03-17</code></td>
      <td class="num">7,836</td>
      <td class="num">2,187</td>
      <td class="num">+5,649 (+258.2%)</td>
      <td class="num">+5.18</td>
      <td>St. Patrick's Day and National Cherry Blossom Festival warm Saturday surge</td>
    </tr>
    <tr>
      <td><code>2012-03-23</code></td>
      <td class="num">8,362</td>
      <td class="num">5,647</td>
      <td class="num">+2,715 (+48.1%)</td>
      <td class="num">+3.42</td>
      <td>Peak Cherry Blossom bloom warm Friday (25&deg;C)</td>
    </tr>
  </tbody>
</table>

{fig9}

<h2>7. Project Codebase Structure and Reproducibility</h2>
<p>
The project is packaged as a self-contained Python repository verified by 24 automated unit, integration, and headless Streamlit tests in <code>tests/test_project.py</code>. Table 6 lists the core modules and their responsibilities.
</p>

<div class="table-caption"><strong>Table 6.</strong> Summary of repository modules and verification scripts.</div>
<table>
  <thead>
    <tr>
      <th>Module / File Path</th>
      <th>Responsibility</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>src/config.py</code></td>
      <td>Central directory paths, commuting-window definitions, feature lists, and UCI category mappings.</td>
    </tr>
    <tr>
      <td><code>src/data_validation.py</code></td>
      <td>Executes 26 automated quality checks on <code>hour.csv</code> and <code>day.csv</code> and writes <code>reports/data_quality_report.md</code>.</td>
    </tr>
    <tr>
      <td><code>src/features.py</code></td>
      <td>Deterministic feature engineering (Fourier harmonics, physical units, rush-hour interactions) and <code>assert_no_leakage()</code>.</td>
    </tr>
    <tr>
      <td><code>src/hypothesis.py</code></td>
      <td>Welch's <i>t</i>-test, Mann-Whitney <i>U</i>, bootstrap CIs, and Newey-West HAC regression (<code>reports/hypothesis_results.md</code>).</td>
    </tr>
    <tr>
      <td><code>src/train_model.py</code> &amp; <code>src/predict.py</code></td>
      <td>Expanding-window CV, model training, residual quantile intervals, and non-negative scenario forecasting.</td>
    </tr>
    <tr>
      <td><code>src/query_engine.py</code> &amp; <code>app.py</code></td>
      <td>Deterministic natural-language intent/entity parser and 7-tab interactive Streamlit operations dashboard.</td>
    </tr>
    <tr>
      <td><code>src/anomaly.py</code></td>
      <td>8-fold block cross-fitted daily residual anomaly detector (<code>outputs/anomalies.csv</code>).</td>
    </tr>
    <tr>
      <td><code>tests/test_project.py</code></td>
      <td>24 automated <code>pytest</code> tests covering data quality, leakage guards, non-negative predictions, and UI rendering.</td>
    </tr>
  </tbody>
</table>

<p>
To reproduce all analytical outputs, run the test suite, or launch the interactive prototype locally, execute the following commands from the project root:
</p>
<pre>
.\\.venv\\Scripts\\python.exe -m src.data_validation
.\\.venv\\Scripts\\python.exe -m src.features
.\\.venv\\Scripts\\python.exe -m src.hypothesis
.\\.venv\\Scripts\\python.exe -m src.train_model
.\\.venv\\Scripts\\python.exe -m src.anomaly
.\\.venv\\Scripts\\python.exe -m pytest -v
.\\.venv\\Scripts\\streamlit.exe run app.py
</pre>

</body>
</html>
"""
    return html


def export_pdf(html_path: Path, pdf_path: Path) -> bool:
    browsers = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    ]
    html_uri = html_path.resolve().as_uri()
    for browser in browsers:
        if not browser.exists():
            continue
        cmd = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path.resolve()}",
            html_uri,
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if pdf_path.exists() and pdf_path.stat().st_size > 10_000:
            return True
    return False


def main() -> None:
    html_content = build_html()

    html_targets = [
        ROOT / "AMAN_KUMAR_RAJ_23CEB0A27_Bike_Sharing_Report.html",
        ROOT / "Complete_Project_Execution_Report.html",
        ROOT / "HR_and_Tech_Submission_Dossier.html",
        config.REPORTS_DIR / "Complete_Project_Execution_Report.html",
        config.REPORTS_DIR / "HR_and_Tech_Submission_Dossier.html",
    ]
    for ht in html_targets:
        ht.write_text(html_content, encoding="utf-8")
        print(f"Saved HTML: {ht} ({ht.stat().st_size:,} bytes)")

    word_html = html_content.replace(
        '<html lang="en">',
        '<html xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:w="urn:schemas-microsoft-com:office:word" '
        'xmlns="http://www.w3.org/TR/REC-html40">',
    ).replace(
        "<head>",
        "<head><!--[if gte mso 9]><xml><w:WordDocument><w:View>Print</w:View>"
        "<w:Zoom>100</w:Zoom><w:DoNotOptimizeForBrowser/></w:WordDocument></xml><![endif]-->",
    )
    for dt in (
        ROOT / "AMAN_KUMAR_RAJ_23CEB0A27_Bike_Sharing_Report.doc",
        ROOT / "HR_and_Tech_Submission_Dossier.doc",
    ):
        try:
            dt.write_text(word_html, encoding="utf-8")
            print(f"Saved DOC:  {dt} ({dt.stat().st_size:,} bytes)")
        except PermissionError:
            pass

    primary_pdf = ROOT / "AMAN_KUMAR_RAJ_23CEB0A27_Bike_Sharing_Report.pdf"
    if export_pdf(html_targets[0], primary_pdf):
        pdf_copies = [
            ROOT / "Complete_Project_Execution_Report.pdf",
            ROOT / "HR_and_Tech_Submission_Dossier.pdf",
            config.REPORTS_DIR / "AMAN_KUMAR_RAJ_23CEB0A27_Bike_Sharing_Report.pdf",
            config.REPORTS_DIR / "Complete_Project_Execution_Report.pdf",
            config.REPORTS_DIR / "HR_and_Tech_Submission_Dossier.pdf",
        ]
        for dst in pdf_copies:
            shutil.copy2(primary_pdf, dst)
            print(f"Saved PDF:  {dst} ({dst.stat().st_size:,} bytes)")
        print(f"Saved Primary PDF: {primary_pdf} ({primary_pdf.stat().st_size:,} bytes)")
    else:
        print("Headless browser PDF export requires running outside the sandbox.")


if __name__ == "__main__":
    main()

