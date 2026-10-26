# How to Open, Run & Explore This Project

This guide provides step-by-step instructions for **HR reviewers**, **technical interviewers**, and **developers** to open the submission documents, launch the interactive Streamlit dashboard, re-run the data/ML pipelines, and execute the automated test suite.

---

## 1. Quickest Way to Review (No Coding or Setup Required)

If you want to review the complete business investigation, methodology, embedded charts, statistical results, and interview talking points immediately without running Python:

1. **Microsoft Word Submission Dossier (`.docx` / `.doc`):**
   - Double-click **`HR_and_Tech_Submission_Dossier.docx`** (or **`HR_and_Tech_Submission_Dossier.doc`**) in the root folder (also available inside `reports/`).
   - Opens in Microsoft Word, Google Docs, or LibreOffice with all 9 charts and tables embedded.
2. **Self-Contained Web Report (`.html`):**
   - Double-click **`reports/HR_and_Tech_Submission_Dossier.html`**.
   - Opens in any web browser (Chrome, Edge, Firefox, Safari) with all charts embedded—no internet connection or server required.
3. **Markdown Reports (in GitHub or VS Code):**
   - `reports/HR_and_Tech_Submission_Dossier.md` — Full technical and executive submission dossier.
   - `reports/final_findings.md` — Summary of empirical findings, limitations, and citations.
   - `reports/data_quality_report.md` — Output of all 26 automated data quality checks.
   - `reports/hypothesis_results.md` — Welch's $t$-test, Mann-Whitney $U$, bootstrap CI, and Newey-West HAC regression results.
   - `reports/model_results.md` — Expanding-window cross-validation and held-out test metrics.
   - `reports/initial_audit.md` — Audit of the legacy R report (`time-series-analysis.html`).

---

## 2. How to Launch the Interactive Streamlit Dashboard & Natural-Language Prototype

### Option A: If You Are Already on This Machine (`c:\Users\Asus\Downloads\bike+sharing+dataset`)

The virtual environment (`.venv`) and trained model (`models/demand_model.joblib`) are already set up. Open **PowerShell** in this project folder and run:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Your browser will automatically open to **`http://localhost:8501`**.

---

### Option B: Cloning Fresh from GitHub (Windows, macOS, or Linux)

#### 1. Clone the repository
```bash
git clone https://github.com/Ghostofficial141/bike-sharing-demand-analytics.git
cd bike-sharing-demand-analytics
```

#### 2. Create and activate a Python virtual environment

- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

- **Windows (Command Prompt / `cmd.exe`):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  pip install -r requirements.txt
  ```

- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

#### 3. Launch the Streamlit application
```bash
streamlit run app.py
```
Then open **`http://localhost:8501`** in your browser.

---

## 3. How to Use the Streamlit Application (`app.py`)

Once `http://localhost:8501` opens in your browser, you will see:

1. **Left Sidebar (Operational Controls):**
   - **Season label convention toggle**: Switch between the official UCI `Readme.txt` labels (`1=spring..4=winter`) and astronomical calendar labels (`1=winter..4=fall`).
   - **Safety planning buffer slider (`0%` to `40%`, default `15%`)**: Adjusts the operational buffer added on top of the model's statistical prediction.
2. **Tab 1 — 💬 Natural-Language Assistant:**
   - Click any of the **9 quick example buttons** at the top, or type your own question in the bottom chat box, such as:
     - *"When is bike demand highest on working days?"*
     - *"Compare demand on working days and non-working days."*
     - *"How does weather affect rental demand?"*
     - *"What is the average demand in summer?"*
     - *"Estimate bike rentals at 8 AM in summer on a working day."*
     - *"Compare expected demand during clear weather and light rain."*
     - *"Which hours should the operations team monitor most closely?"*
     - *"Was the working-day difference statistically significant?"*
     - *"Which days had unusual demand?"*
   - Works **100% offline** with no API key required.
3. **Tab 2 — 📊 Demand Estimator (What-If Simulator):**
   - Use interactive sliders for Season, Hour of Day (`0–23`), Day Classification (`Working day`, `Weekend`, `Public holiday`), Weather Category (`1–4`), Temperature (`°C`), Humidity (`%`), Wind Speed (`km/h`), and Year regime (`2012` vs `2011`).
   - Displays the **Expected Rentals/hr**, **Indicative 80% Range**, **Baseline Estimate**, **Planning Target (with safety buffer)**, and a **24-hour bar chart** highlighting your chosen hour.
4. **Tabs 3 to 7:**
   - **📈 Hourly, Seasonal & Day-Type Patterns**
   - **🌦️ Weather & Temperature Impact**
   - **🧪 Hypothesis Testing**
   - **🤖 Model Performance & Diagnostics**
   - **🚨 Anomaly & Event Detection**

---

## 4. How to Re-Run Each Stage of the Pipeline from Scratch

All pre-computed outputs, figures, reports, and model bundles are already included in the repository. If you want to regenerate everything from scratch, run these commands in order (using `.\.venv\Scripts\python.exe` on Windows or `python` when `.venv` is activated):

```powershell
# Step 1: Verify (or download) the official UCI dataset in data/raw/
.\.venv\Scripts\python.exe scripts/download_data.py

# Step 2: Run data quality validation -> writes reports/data_quality_report.md & outputs/data_quality.json
.\.venv\Scripts\python.exe -m src.data_validation

# Step 3: Build engineered features -> writes data/processed/hour_features.csv
.\.venv\Scripts\python.exe -m src.features

# Step 4: Generate exploratory & data-completeness plots -> writes outputs/figures/eda_*.png
.\.venv\Scripts\python.exe -m src.eda_plots

# Step 5: Run statistical hypothesis tests -> writes reports/hypothesis_results.md & outputs/hypothesis_results.json
.\.venv\Scripts\python.exe -m src.hypothesis

# Step 6: Train & evaluate the demand-estimation models -> writes models/demand_model.joblib & reports/model_results.md
.\.venv\Scripts\python.exe -m src.train_model

# Step 7: Run conditional anomaly detection -> writes outputs/anomalies.csv & outputs/figures/anomaly_detection.png
.\.venv\Scripts\python.exe -m src.anomaly

# Step 8: Rebuild the HTML, Markdown, and Word (.doc / .docx) submission dossiers
.\.venv\Scripts\python.exe scripts/build_submission_dossier.py
.\.venv\Scripts\python.exe scripts/build_word_dossier.py
```

---

## 5. How to Run the Automated Test Suite (`pytest`)

To execute all **24 automated unit and integration tests** (covering data validation, target leakage guards, chronological splits, nonnegative predictions, hypothesis tests, all natural-language query intents, anomaly detection, and headless Streamlit UI rendering):

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Expected output:
```text
============================= 24 passed in ~17s =============================
```

---

## 6. Troubleshooting

- **PowerShell script execution policy error when running `Activate.ps1`:**
  - Either run commands directly using `.\.venv\Scripts\python.exe ...` and `.\.venv\Scripts\streamlit.exe run app.py` (which bypasses `Activate.ps1` completely), or run:
    ```powershell
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1
    ```
- **Port 8501 is already in use:**
  - Launch Streamlit on another port:
    ```powershell
    .\.venv\Scripts\streamlit.exe run app.py --server.port 8502
    ```
- **Matplotlib cache warning on Windows:**
  - Set `$env:MPLCONFIGDIR="$PWD\.mplcache"` before running plotting scripts if your Windows user profile restricts writing to `C:\Users\<user>\.matplotlib`.
