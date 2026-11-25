# Project Analysis

Repository: `Prem954/Software-Requirement-Specification-Document-Analysis-and-Parameter-Identification-Using-ML` (branch `main`, HEAD `9768a62`).
Every statement below comes from reading the source code and running it. Where the README and the code disagree, this is noted.

## 1. Problem

Reading SRS (Software Requirement Specification) documents by hand to find which requirements relate to which evaluation parameter is slow and error-prone. The project targets SRS documents that describe a **cloud service provider**.

## 2. Proposed solution

A supervised text classifier assigns every sentence of an SRS PDF to one of 8 predefined **parameters**:

| # | Parameter (class) | Training sentences |
|---|---|---|
| 0 | Availability | 10 |
| 1 | Compatibility with Microsoft Products | 10 |
| 2 | Disaster Recovery | 10 |
| 3 | Establishment | 24 |
| 4 | Market Share | 10 |
| 5 | Offline Data Transfer | 10 |
| 6 | Pricing Models | 10 |
| 7 | Storage Service Options | 10 |

The repository also has a separate retrieval-augmented chatbot (`srs-chatbot/`) that answers questions about an SRS PDF using Google Gemini.

## 3. Machine learning algorithms (verified)

| Component | Implementation | Source |
|---|---|---|
| Text vectorisation | `TfidfVectorizer()`, default settings (unigrams, lowercase); 340-term vocabulary after fitting | `final.py`, the pickle |
| Classifier | `LinearSVC()`, a linear Support Vector Machine using one-vs-rest, default `C=1.0` | `final.py`, the pickle |
| Label encoding | `LabelEncoder` | `label_encoder.pkl` |
| Pipeline | `make_pipeline(TfidfVectorizer(), LinearSVC())` saved with joblib using **scikit-learn 1.5.2** | `sentence_classifier_pipeline.pkl` |
| Chatbot (separate) | `all-MiniLM-L6-v2` sentence embeddings + FAISS similarity search + Gemini `gemini-pro` LLM | `srs-chatbot/app.py` |

No hyperparameter tuning, cross-validation or probability calibration is done. Because `LinearSVC` produces no probabilities, the original code has **no confidence score**.

**Reported accuracy:** the notebook's output shows **78.95 %** on a 19-sentence test split. Re-running the saved model locally on the same split (`evaluate_model.py`) gives exactly **78.95 %**, with an identical classification report. Macro F1 is 0.69 and Market Share recall is 0.00 (1 test sample). With only 19 test sentences, this figure is very noisy.

## 4. How document analysis works (`final.py`, part 2)

1. `fitz.open(pdf)` → `page.get_text()` for each page, joined into one string (PyMuPDF).
2. `preprocess_text()` splits the text with the regex `(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s`, i.e. at whitespace following `.` or `?`, while avoiding splits after abbreviations like `e.g.` or `Mr.`.
3. Each sentence → `pipeline.predict([sentence])` → `label_encoder.inverse_transform()` → printed.

No stop-word removal, lemmatisation, OCR or section detection is done. Headings and list numbers ("2.", "3.") come out as separate "sentences" and still get a label.

## 5. How parameters are identified

A parameter counts as "identified" when the classifier assigns at least one sentence to it. The model has **no "none / other" class**, so every sentence is forced into one of the 8 parameters. That includes non-functional requirements (security, usability), titles and conclusions. Parameter *values* (e.g. the number "99.99 %") are **not** extracted; only the category is predicted.

## 6. Dataset

`Dataset.csv` has columns `Sentence,Label` and 95 rows. One row is empty, so 94 remain after `dropna()`. The sentences are short, hand-written statements about cloud providers. The classes are imbalanced: Establishment has 24 sentences and every other class has 10. The data comes only from the cloud-provider domain, so the model does not generalise to other kinds of SRS without new training data.

Bundled sample: `SRS.pdf` (4 pages, a cloud-provider SRS with one section per parameter; also copied as `srs-chatbot/pdf/SRS.pdf`). Only 1 of its 57 extracted sentences appears verbatim in the training data.

## 7. Technologies

| Layer | Original repository | Added in this local setup |
|---|---|---|
| Language | Python (Colab 3.10; chatbot venv 3.12) | Python 3.9.4 venv |
| ML | scikit-learn, pandas, joblib | same, scikit-learn pinned to 1.5.2 |
| PDF | PyMuPDF (classifier); `PyPDFLoader` / pypdf (chatbot) | PyMuPDF 1.24.13 |
| Web | Flask + Flask-Session, HTML/CSS/JS (chatbot only) | Flask 3.1 dashboard |
| LLM / RAG | LangChain, HuggingFace embeddings, FAISS, Google Gemini | not installed (optional) |
| Database | **none**: no database integration exists; the chatbot uses filesystem sessions | JSON report files |

## 8. Modules

| File | Role | Notes |
|---|---|---|
| `Final.ipynb` / `final.py` | Training (part 1) and PDF classification (part 2) | Colab-only: `google.colab.files.upload()`, `!pip`. `final.py` is not valid Python because of the `!pip` line. |
| `Dataset.csv` | Training data | 94 labelled sentences |
| `sentence_classifier_pipeline.pkl`, `label_encoder.pkl` | Trained artefacts | sklearn 1.5.2 |
| `srs-chatbot/app.py` | Flask RAG chatbot over `srs-chatbot/pdf/*.pdf` | Does **not** use the classifier. Hard-coded placeholder API key `"Your_API_Key"`; `gemini-pro` is a retired model name. |
| `srs-chatbot/templates/index.html`, `static/` | Chat UI | Plain chat window, no upload |
| `srs-chatbot/myenv/` | Committed virtualenv | Contains only pip, with paths from the author's machine; unusable |
| `srs-chatbot/flask_session/` | Committed session files | Runtime leftovers |
| `Code simple explanation for Srs analysis.pdf` | Step-by-step explanation of `final.py` | Matches the code |
| `Workflow.svg` | Workflow diagram (image only) | |

**README vs code:** the README says that uploaded PDFs are classified "in real time" through the Flask app and that the chatbot explains classified parameters. In the code, the Flask app never loads the classifier, has no upload feature, and answers only from PDFs in the `pdf/` folder. The two parts are unconnected.

## 9. Complete workflow

```
Dataset.csv ─► dropna ─► LabelEncoder ─► train_test_split(80/20, rs=42)
           ─► TF-IDF + LinearSVC ─► evaluate (accuracy, report) ─► joblib.dump(.pkl)

SRS PDF ─► PyMuPDF text ─► regex sentence split ─► pipeline.predict ─► inverse_transform
        ─► (sentence, parameter) list
                                            [added locally] ─► dashboard table/charts/CSV/JSON

(separate) srs-chatbot: pdf/*.pdf ─► PyPDFLoader chunks ─► MiniLM embeddings ─► FAISS
        user question ─► similarity_search ─► Gemini prompt ─► answer
```

## 10. Limitations found

- Very small dataset (94 sentences) from one domain; the test set has only 19 sentences.
- No "other" class, so off-topic sentences still receive a parameter.
- The sentence splitter produces heading and numbering fragments and merges headings with the sentence that follows.
- No confidence or probability output in the original method.
- Only text-based PDFs are supported: no OCR and no DOCX.
- The chatbot depends on a paid external API, a retired model name and a heavy ML stack, and it is not connected to the classifier.
