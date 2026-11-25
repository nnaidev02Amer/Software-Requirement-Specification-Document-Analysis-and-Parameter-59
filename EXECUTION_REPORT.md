# Execution Report

Date: 2026-11-25 · Machine: Windows 11, 8 GB RAM · Runtime: Python 3.9.4 `.venv`

## 1. Clone

```
git clone https://github.com/Prem954/Software-Requirement-Specification-Document-Analysis-and-Parameter-Identification-Using-ML.git
→ branch main, HEAD 9768a62 "Update LICENSE", 1055 files, 25 MB
```
Path: `C:\Users\SURESH\Desktop\kits project\Software-Requirement-Specification-Document-Analysis-and-Parameter-Identification-Using-ML`

## 2. Environment

| Item | Result |
|---|---|
| Global Python | 3.14.6 (untouched). Not used because scikit-learn 1.5.2 does not support it |
| Project venv | `.venv` on Python 3.9.4 |
| Installed | scikit-learn 1.5.2, numpy 1.26.4, scipy 1.13.1, pandas 2.2.3, PyMuPDF 1.24.13, Flask 3.1.3, joblib 1.5.3 |
| Install issue | First attempt failed with `ReadTimeoutError` (network); fixed with `--timeout 120 --retries 10` |
| Model compatibility | The pickles record sklearn 1.5.2, matching the runtime; they load with no warnings |

## 3. Component status

| Check | Status | Evidence |
|---|---|---|
| Dataset loading | ✅ | 95 rows → 94 after `dropna`, 8 classes |
| Model loading | ✅ | `TfidfVectorizer → LinearSVC`, 340-term vocabulary |
| Saved-model accuracy | ✅ **78.95 %** | `evaluate_model.py`; matches the notebook output exactly (19 test sentences) |
| PDF upload (HTTP multipart) | ✅ | `curl -F document=@SRS.pdf http://127.0.0.1:5000/analyze` → 302 to the report |
| Text extraction / sentence split | ✅ | SRS.pdf: 4 pages, 57 sentences |
| Classification | ✅ | All 57 sentence/label pairs are identical to the original Colab output (automated test) |
| Report / CSV / JSON export | ✅ | `/report/<id>`, `/download.csv`, `/download.json` |
| Dashboard startup | ✅ | `python dashboard/app.py` → http://127.0.0.1:5000 |
| Original chatbot (`srs-chatbot/app.py`) | ⚠️ Not run | Syntax OK. Needs about 1.5–2 GB of extra packages (PyTorch) and a Gemini API key, and uses the retired `gemini-pro` model. See SETUP_GUIDE §7 |
| Original `final.py` | ⚠️ Not runnable as-is | Colab-only (`google.colab`, `!pip`). Replaced locally by `run_analysis.py` / `evaluate_model.py` with the same logic |

## 4. Tests: `python -m unittest discover -s tests -v`

13 run, **13 passed**, 0 failed.

```
test_pages / test_rejects_non_pdf / test_unknown_report_404 / test_upload_pdf_end_to_end   (dashboard)
test_dataset / test_imports / test_required_files_exist / test_saved_version_matches_runtime  (files & deps)
test_pdf_matches_original_notebook_output / test_pipeline_structure / test_preprocess_text /
test_reproduces_notebook_accuracy / test_training_sentences_classified                    (model)
Ran 13 tests ... OK
```
The only warnings are harmless SWIG `DeprecationWarning`s from PyMuPDF on Python 3.9.

## 5. Actual analysis output

### 5a. `SRS.pdf` (bundled sample), uploaded through the dashboard

4 pages · 57 sentences · 53 requirement sentences (4 numbering fragments) · 8/8 parameters identified.

| Parameter | Sentences |
|---|---|
| Availability | 7 |
| Compatibility with Microsoft Products | 7 |
| Disaster Recovery | 7 |
| Establishment | 7 |
| Market Share | 4 |
| Offline Data Transfer | 5 |
| Pricing Models | 7 |
| Storage Service Options | 9 |

Excerpt (SVM score = LinearSVC decision margin for the predicted class, **not** a probability):

| # | Sentence (shortened) | Predicted | SVM score |
|---|---|---|---|
| 13 | A 99.99% uptime guarantee ensures continuous service. | Availability | 0.239 |
| 18 | Pay-as-you-go and subscription-based models provide customers with flexible payment options… | Pricing Models | 0.398 |
| 23 | Hardware devices are provided for secure offline data transfer… | Offline Data Transfer | 0.644 |
| 28 | Integration with Microsoft Azure Active Directory and Microsoft SQL Server is fully supported. | Compatibility with Microsoft Products | 0.870 |
| 33 | Regular automatic backups ensure data can be restored in case of emergencies. | Disaster Recovery | 0.117 |
| 39 | The provider's market growth outpaces many competitors… | **Establishment** (sits in the Market Share section) | 0.499 |
| 43 | Hot, cold, and archival storage tiers are available… | Storage Service Options | 0.539 |
| 50 | The system should comply with industry-standard security protocols… | Storage Service Options (no "security" class exists) | -0.467 |

Manual check against the PDF's section headings: in sections 3.1–3.8, **36 of 38** sentences got the label of the section they sit in. The two exceptions are #10 (*market presence* → Market Share, in the Establishment section) and #39 (above). This is my reading of the headings, not a labelled ground truth.

### 5b. `samples/sample_cloud_storage_srs.pdf`: new text, written for this demo, not in the training data

1 page · 23 sentences · 16 requirement sentences · 8/8 parameters. Of the 14 sentences written for a specific parameter section, **13** got that section's label. The miss: *"Users should be able to open Word and Excel files directly from storage."* → Storage Service Options, where Compatibility with Microsoft Products was intended. The title and purpose sentences were forced into Storage Service Options / Establishment because the model has no "other" class.

Full outputs can be downloaded from each report page as CSV or JSON.

## 6. Screenshots (`docs/screenshots/`)

| File | Shows |
|---|---|
| `01_home.png` | Upload page, parameter list, recent analyses |
| `02_report_srs_pdf.png` | Full report for SRS.pdf |
| `03_report_sample_unseen.png` | Full report for the unseen sample PDF |
| `04_model_dataset.png` | Dataset distribution, confusion matrix, classification report |

Captured with headless Microsoft Edge against the running server.

## 7. Fixes and additions (original files unchanged)

| Problem | Action |
|---|---|
| No requirements file | Added `requirements.txt` (verified) and `requirements-chatbot.txt` (optional, unverified) |
| Classifier only runs in Colab | Added `srs_classifier.py` (same regex, the same saved pickles loaded unchanged), `run_analysis.py`, `evaluate_model.py` |
| Flask app does not use the ML model, so there is no upload or classification UI | Added the `dashboard/` Flask app: upload, progress indicator, results table, charts, CSV/JSON/print export, model page |
| No tests | Added `tests/test_project.py` (13 tests) |
| Pickle version risk | Pinned scikit-learn 1.5.2 on Python 3.9 |
| pip read timeouts | Retries and a longer timeout |
| Venv and runtime files would be committed | Added `.gitignore` (`.venv/`, uploads, reports) |

No model was retrained. Nothing was committed or pushed.

## 8. Remaining issues / limitations

1. The original Gemini chatbot was not executed (API key, retired model name, heavy dependencies).
2. 94-sentence, single-domain dataset; 78.95 % accuracy on just 19 test sentences.
3. No "other" class, so titles and non-functional requirements are forced into one of 8 parameters.
4. The regex splitter leaves numbering fragments ("2.") and merges headings into sentences. The dashboard flags fragments of 3 words or fewer and hides them by default.
5. PDF only (no DOCX, no OCR for scanned documents).
6. The progress steps on the upload page are a visual indicator; analysis runs as one synchronous request (it takes under 1 s for these PDFs).
7. `dashboard/reports/` contains one extra SRS.pdf report created by the first test run, before the tests were changed to use a temporary folder. It is a real analysis and can be deleted.

---

## 9. Dashboard redesign: "SRS Intelligence" (2026-11-25)

The Flask dashboard was redesigned into a sidebar-based analytics app. The ML pipeline, PDF extraction, sentence splitting and saved model are unchanged, and nothing was retrained.

**Pages and routes**

| Sidebar item | Route | Content (all computed from stored reports, the saved model or Dataset.csv) |
|---|---|---|
| Dashboard | `/` | Welcome banner, 4 KPI cards, upload card, sentences-per-parameter bars, parameter-distribution donut, recent activity, recent documents, ML overview |
| Upload SRS Document | `/upload` | Drag-and-drop or browse, real XHR upload progress, analysis steps |
| My Documents | `/documents?q=` | All reports with search, status, View / CSV / JSON |
| SRS Analysis | `/analysis` → `/report/<id>` | Latest report (document switcher), sentence table, exports, print |
| Identified Parameters | `/parameters?doc=` | 8 parameter cards and sentences per parameter, for all documents or one |
| Analytics & Reports | `/analytics` | Aggregate charts, document × parameter heatmap, summary CSV, per-document downloads |
| ML Model Performance | `/model` | Accuracy 78.95 %, macro F1, pipeline, confusion matrix, per-class metrics, sklearn report |
| Settings | `/settings` | Display name and sidebar/fragment preferences (stored in browser localStorage), live system information |

Kept unchanged: `POST /analyze`, `/report/<id>/download.csv`, `/report/<id>/download.json`, `/health`. Added: `/reports/summary.csv`.

**Honesty notes:** The app has no user accounts. The profile name defaults to "Project User" and can be changed in Settings; it is stored only in the browser. Notifications list the most recent real analyses. Icons come from Lucide 0.460.0 (ISC licence), vendored in `dashboard/static/vendor/`, so no CDN or Python package was added.

**Verification**
- 17/17 unit and integration tests pass. The 4 new tests check that every sidebar page renders with an active nav item, that dashboard numbers match the backend aggregates, the summary CSV, and the `/analysis` redirect.
- All routes return HTTP 200, and a missing report returns 404.
- Real browser upload: headless Edge attached a PDF, the XHR upload ran, and the browser landed on the new report page. The test report was then removed.
- Screenshots in `docs/screenshots/` (`v2_01`…`v2_12`): desktop pages, 390 px mobile layout, mobile off-canvas menu, collapsed sidebar, browser upload result.
