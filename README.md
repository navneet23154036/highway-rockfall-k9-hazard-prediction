# Highway Rockfall k=9 Hazard Prediction

**Updated version — v4 k=9 Gradient Boosting + Ridge ensemble**

[Live Website](https://highway-rockfall-k9-hazard-prediction.onrender.com) · [GitHub Repository](https://github.com/navneet23154036/highway-rockfall-k9-hazard-prediction)

## Overview

Highway Rockfall k=9 Hazard Prediction is an updated web application for **score-based highway slope hazard assessment**. It combines slope geometry, site information and inspector-assigned environmental, traffic and geological ratings to estimate a **predicted total hazard score**. The score is compared with applicable classification cutoffs to assign **Low**, **Moderate** or **High** hazard.

The updated model is described as a **v4 k=9 Gradient Boosting + Ridge ensemble**. The application also supports input guidance, saved assessment history and downloadable PDF reports.

## Key Features

- **Updated ML pipeline:** k=9 ensemble-based hazard score prediction.
- **Three hazard classes:** Low, Moderate and High, based on score cutoffs.
- **Interpretability:** displays predicted score, classification ranges and distance from a cutoff.
- **Guided inputs:** human-readable labels and an *About inputs* guide.
- **Example presets:** Low-score and High-score example input controls for demonstration.
- **Assessment history:** save and review assessments in the web interface.
- **PDF reports:** downloadable report containing assessment scores, hazard class distribution, submitted parameters and charts.
- **Branded reporting:** project logo/watermark styling in PDF exports.
- **Responsive interface:** HTML, CSS and JavaScript frontend with a Flask backend.

## How Hazard Classification Works

The backend returns a predicted total score and the applicable **Low cutoff** and **High cutoff**. The class is determined by:

| Hazard class | Rule |
| --- | --- |
| **Low** | Predicted score ≤ Low cutoff |
| **Moderate** | Low cutoff < Predicted score ≤ High cutoff |
| **High** | Predicted score > High cutoff |

### Example (illustrative only)

Suppose the predicted score is **815.5**, with a Low cutoff of **445.2** and High cutoff of **595.8**. Since `815.5 > 595.8`, the assessment is classified as **High**. The score is **219.7 points above the High cutoff**.

**Important:** The predicted score is **not a percentage or calibrated probability of a future rockfall**. A difference from a cutoff is measured in score points, not as a percentage increase in hazard. Actual cutoffs should always come from the backend response for the assessment.

## Input Categories

The assessment form includes:

1. **Site and geology:** assessment name, rock type, slope type and engineering region.
2. **Slope geometry:** slope height, angle and length, plus ditch slope, width and depth.
3. **Inspector ratings:** rockfall frequency, precipitation, freeze–thaw activity, seepage/water, aspect, slope and ditch scores, traffic-related ratings and rock-specific geological ratings.

Inspector scores should be selected using the appropriate CDOT assessment criteria rather than guessed. The interface's *About inputs* guide explains the fields.

## Technology Stack

| Layer | Technology |
| --- | --- |
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| Machine learning | Gradient Boosting + Ridge ensemble (v4 k=9) |
| Data processing / model loading | pandas, joblib, scikit-learn |
| PDF reporting | ReportLab |
| Deployment | Render |

## Main Project Files

```text
highway-rockfall-k9-hazard-prediction/
├── app.py                        # Flask app and prediction endpoints
├── app.js                        # UI interactions, assessments and API calls
├── index.html                    # Main website
├── style.css                     # Styling
├── pdf_report.py                 # Server-side PDF report generation
├── report.js                     # Reporting-related frontend logic
├── rockfall_v4_k9.joblib         # Exported model bundle (required)
├── rockfall_logo.png             # Project logo (if included)
├── notebook_source.py            # Model notebook-derived source
├── train_export.py               # Model export utility
├── slope-measurement.gif         # Geometry guide illustration
├── requirements.txt              # Python dependencies
└── README.md
```

> File names can vary between releases. Confirm the exact model and logo filenames expected by `app.py` and `pdf_report.py` before deployment. Do not commit virtual environments or Python cache folders.

## Run Locally

**Requirements:** Python 3 and pip. The trained `.joblib` model bundle must be available at the path expected by the backend.

```bash
git clone https://github.com/navneet23154036/highway-rockfall-k9-hazard-prediction.git
cd highway-rockfall-k9-hazard-prediction
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

For troubleshooting, check these backend endpoints when supported by your version:

- `GET /health` — application/model readiness.
- `GET /schema` — expected model input fields.
- `POST /predict` — hazard assessment prediction.

The application must load its model successfully for the assessment fields and predictions to work.

## Generate an Assessment and Report

1. Open **Slope assessment**.
2. Enter site details and slope geometry.
3. Select the inspector ratings according to the input guide.
4. Click **Assess slope** to obtain the predicted score and hazard class.
5. Save the assessment to **Assessment history**.
6. Download the PDF report to review the results, submitted input parameters, charts and classification explanation.

The example preset buttons are intended for **demonstration**, not for replacing actual field measurements.

## Deployment on Render

Create a **separate** Render Python web service linked to this repository:

- **Branch:** `main`
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app`

Ensure the deployed repository contains the required model bundle, the dependencies needed to load it and any PDF branding assets. After deployment, verify `/health`, `/schema`, an actual prediction and PDF export.

## Previous vs Updated Version

| Feature | Previous version | Updated k=9 version |
| --- | --- | --- |
| Model | Random Forest | Gradient Boosting + Ridge ensemble |
| Result | Hazard classification | Predicted total score + hazard class |
| Interpretation | Basic result | Cutoff ranges and score-point difference |
| Inputs | Original assessment form | Guided fields and example presets |
| Reporting | Earlier reporting workflow | Assessment history and detailed PDF reporting |

[View Previous Website](https://highway-rockfall-hazard-classification.onrender.com)

## Limitations and Responsible Use

- The model estimates a **hazard score**, not the probability of an imminent rockfall event.
- Assessment quality depends on accurate geometry and properly assigned inspector ratings.
- Scores from different slope segments should not be interpreted as changes over time at one location.
- The application is a research and screening tool; it does **not** replace a qualified geotechnical assessment or field verification.

## Developer

**Navneet Gupta**

---

*Final-year project: machine-learning-based highway rockfall hazard assessment.*
