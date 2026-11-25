# Setup Guide

Verified on Windows 11 with Python 3.9.4 and 8 GB RAM. The core setup downloads about 120 MB and the running dashboard uses about 150 MB of RAM.

## 1. Prerequisites

- Git
- Python **3.9 – 3.12**. scikit-learn 1.5.2, which the saved model needs, has no wheels for Python 3.13+.
  Check with `py -0` (Windows) or `python3 --version`.

## 2. Clone

```powershell
cd "C:\Users\SURESH\Desktop\kits project"
git clone https://github.com/Prem954/Software-Requirement-Specification-Document-Analysis-and-Parameter-Identification-Using-ML.git
cd Software-Requirement-Specification-Document-Analysis-and-Parameter-Identification-Using-ML
```

## 3. Virtual environment and dependencies

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt
```

`--no-cache-dir` keeps disk and RAM use low. On a slow connection, `--timeout`/`--retries` avoid the read timeouts seen during setup.

> Do **not** use `srs-chatbot/myenv`. It is a broken virtualenv committed from the author's PC.

## 4. Run the dashboard (main application)

```powershell
.\.venv\Scripts\python.exe dashboard\app.py
```

Open **http://127.0.0.1:5000**.

- **Analyse document**: upload a PDF, or click *Use bundled sample (SRS.pdf)*.
- The report page shows sentence counts, a chart of sentences per parameter, the identified parameters, a filterable table, and CSV / JSON / print-to-PDF export.
- **Model & dataset**: training data distribution, plus the saved model's test accuracy, confusion matrix and classification report, computed live.

Optional environment variables: `SRS_HOST`, `SRS_PORT`, `SRS_DATA_DIR` (where uploads and reports are stored; default `dashboard/`).

## 5. Command-line equivalents

```powershell
.\.venv\Scripts\python.exe run_analysis.py                    # classify SRS.pdf (same output as the notebook)
.\.venv\Scripts\python.exe run_analysis.py my.pdf --csv out.csv
.\.venv\Scripts\python.exe evaluate_model.py                  # accuracy of the saved model on the notebook's test split
.\.venv\Scripts\python.exe samples\make_sample_pdf.py         # regenerate the unseen demo PDF
```

## 6. Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 7. Optional: original Gemini chatbot (not verified)

`srs-chatbot/app.py` needs PyTorch, sentence-transformers and FAISS (about 1.5–2 GB, which is heavy on 8 GB RAM) plus a Google AI Studio API key.

```powershell
py -3.11 -m venv .venv-chatbot
.\.venv-chatbot\Scripts\python.exe -m pip install -r requirements-chatbot.txt
```

Before running it, edit `srs-chatbot/app.py`:
1. Replace `"Your_API_Key"` with your key, or better, read it from an environment variable.
2. Change `model="gemini-pro"` to a model name that is currently available (`gemini-pro` has been retired).

Then run `cd srs-chatbot; ..\.venv-chatbot\Scripts\python.exe app.py`. It must be started from inside `srs-chatbot/` because it reads the relative `pdf/` folder.

## Troubleshooting

| Problem | Fix |
|---|---|
| `InconsistentVersionWarning` or unpickling error | You are not on scikit-learn 1.5.2; recreate `.venv` with Python 3.9–3.12 |
| `ReadTimeoutError` during pip install | Re-run with `--timeout 120 --retries 10` |
| "No text could be extracted" | The PDF is a scanned image; OCR is not supported |
| Port 5000 in use | `$env:SRS_PORT=5050; .\.venv\Scripts\python.exe dashboard\app.py` |
