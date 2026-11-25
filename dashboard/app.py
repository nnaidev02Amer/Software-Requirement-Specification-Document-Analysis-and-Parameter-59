"""SRS Intelligence dashboard: upload an SRS PDF and classify its sentences.

Uses the project's saved TF-IDF + LinearSVC model through srs_classifier.py.
Every number shown in the UI is computed from the saved model, Dataset.csv
or the analysis reports stored on disk - nothing is hard-coded.

Run from the project root:   python dashboard/app.py
"""
import csv
import io
import json
import os
import sys
import uuid
from collections import Counter, OrderedDict
from datetime import datetime

from flask import (Flask, Response, abort, flash, redirect, render_template,
                   request, url_for)
from werkzeug.utils import secure_filename

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from srs_classifier import (DATASET_PATH, FRAGMENT_MAX_WORDS, SAMPLE_PDF_PATH,  # noqa: E402
                            extract_pdf_text, get_classifier, pdf_page_count,
                            preprocess_text)

DATA_DIR = os.environ.get("SRS_DATA_DIR", HERE)
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
REPORT_DIR = os.path.join(DATA_DIR, "reports")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

MAX_UPLOAD_MB = 10
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["SECRET_KEY"] = os.environ.get("SRS_SECRET_KEY", uuid.uuid4().hex)

# Fixed categorical order (validated palette, see dashboard/static/dashboard.css).
PARAM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#6250d6", "#e34948"]
PARAM_ICONS = {
    "Availability": "activity",
    "Compatibility with Microsoft Products": "layers",
    "Disaster Recovery": "shield-check",
    "Establishment": "building-2",
    "Market Share": "chart-pie",
    "Offline Data Transfer": "hard-drive-download",
    "Pricing Models": "badge-dollar-sign",
    "Storage Service Options": "database",
}

_eval_cache = {}


# --------------------------------------------------------------------------- data
def _report_path(rid):
    if not rid.isalnum():
        abort(404)
    return os.path.join(REPORT_DIR, f"{rid}.json")


def _load_report(rid):
    path = _report_path(rid)
    if not os.path.exists(path):
        abort(404)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_analysis(pdf_path, original_name):
    clf = get_classifier()
    text = extract_pdf_text(pdf_path)
    sentences = preprocess_text(text)
    preds = [p.to_dict() for p in clf.classify(sentences)]
    requirements = [p for p in preds if not p["is_fragment"]]
    counts = Counter(p["label"] for p in requirements)
    by_label = OrderedDict()
    for label in clf.labels:
        items = [p for p in requirements if p["label"] == label]
        if items:
            by_label[label] = items
    rid = uuid.uuid4().hex[:12]
    report = {
        "id": rid,
        "file_name": original_name,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pages": pdf_page_count(pdf_path),
        "characters": len(text),
        "total_sentences": len(preds),
        "requirement_sentences": len(requirements),
        "fragments": len(preds) - len(requirements),
        "label_counts": {label: counts.get(label, 0) for label in clf.labels},
        "parameters_found": len(by_label),
        "by_label": by_label,
        "predictions": preds,
    }
    with open(_report_path(rid), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def all_reports():
    """Every stored report, newest first."""
    items = []
    for name in os.listdir(REPORT_DIR):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(REPORT_DIR, name), encoding="utf-8") as f:
                r = json.load(f)
            r["status"] = "Completed" if r.get("total_sentences") else "No text found"
            items.append(r)
        except (OSError, ValueError):
            continue
    items.sort(key=lambda r: r.get("created", ""), reverse=True)
    return items


def recent_reports(limit=8):
    return all_reports()[:limit]


def aggregate(reports):
    labels = get_classifier().labels
    totals = Counter()
    docs_with = Counter()
    for r in reports:
        for label, c in r.get("label_counts", {}).items():
            totals[label] += c
            if c:
                docs_with[label] += 1
    req_total = sum(totals.values())
    return {
        "documents": len(reports),
        "pages": sum(r.get("pages", 0) for r in reports),
        "sentences": sum(r.get("total_sentences", 0) for r in reports),
        "requirements": req_total,
        "fragments": sum(r.get("fragments", 0) for r in reports),
        "label_totals": {l: totals.get(l, 0) for l in labels},
        "docs_with": {l: docs_with.get(l, 0) for l in labels},
        "params_identified": sum(1 for l in labels if totals.get(l)),
        "n_labels": len(labels),
        "donut": donut_segments({l: totals.get(l, 0) for l in labels}),
    }


def donut_segments(counts, gap=1.2):
    """SVG stroke-dasharray segments (circumference 100) with small surface gaps."""
    total = sum(counts.values())
    segs, offset = [], 0.0
    for i, (label, c) in enumerate(counts.items()):
        share = (c / total * 100) if total else 0
        if c:
            length = max(share - gap, 0.4)
            segs.append({"label": label, "count": c, "pct": share, "color": PARAM_COLORS[i],
                         "dash": f"{length:.3f} {100 - length:.3f}", "offset": f"{-offset:.3f}"})
        offset += share
    return segs


def evaluation():
    if "data" not in _eval_cache:
        import pandas as pd
        from evaluate_model import evaluate

        r = evaluate()
        data = pd.read_csv(DATASET_PATH).dropna()
        dist = data["Label"].value_counts()
        _eval_cache["data"] = {
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "per_class": r["per_class"],
            "n_rows": r["n_rows"],
            "n_test": r["n_test"],
            "n_train": r["n_rows"] - r["n_test"],
            "report": r["report"],
            "labels": r["labels"],
            "cm": r["confusion_matrix"].tolist(),
            "cm_max": int(r["confusion_matrix"].max()) or 1,
            "correct": int(r["confusion_matrix"].trace()),
            "distribution": {k: int(dist.get(k, 0)) for k in r["labels"]},
        }
    return _eval_cache["data"]


@app.context_processor
def inject_globals():
    labels = get_classifier().labels
    return {
        "model_info": get_classifier().describe(),
        "param_color": {l: PARAM_COLORS[i] for i, l in enumerate(labels)},
        "param_icon": PARAM_ICONS,
        "notifications": recent_reports(5),
        "nav": NAV,
    }


NAV = [
    ("dashboard", "Dashboard", "layout-dashboard", "index"),
    ("upload", "Upload SRS Document", "cloud-upload", "upload"),
    ("documents", "My Documents", "folder-open", "documents"),
    ("analysis", "SRS Analysis", "file-search", "analysis"),
    ("parameters", "Identified Parameters", "tags", "parameters"),
    ("analytics", "Analytics & Reports", "chart-column", "analytics"),
    ("model", "ML Model Performance", "brain-circuit", "model"),
    ("settings", "Settings", "settings", "settings"),
]


# ------------------------------------------------------------------------- pages
@app.route("/")
def index():
    reports = all_reports()
    return render_template("index.html", active="dashboard", recent=reports[:6],
                           agg=aggregate(reports), ev=evaluation())


@app.route("/upload")
def upload():
    return render_template("upload.html", active="upload", recent=recent_reports(5))


@app.route("/documents")
def documents():
    q = request.args.get("q", "").strip()
    reports = all_reports()
    if q:
        reports = [r for r in reports if q.lower() in r["file_name"].lower()]
    return render_template("documents.html", active="documents", docs=reports, q=q)


@app.route("/analysis")
def analysis():
    reports = all_reports()
    if not reports:
        return render_template("empty.html", active="analysis", title="SRS Analysis",
                               icon="file-search",
                               message="No document has been analysed yet.")
    return redirect(url_for("report", rid=reports[0]["id"]))


@app.route("/report/<rid>")
def report(rid):
    docs = [{"id": d["id"], "file_name": d["file_name"], "created": d["created"]}
            for d in all_reports()]
    return render_template("report.html", r=_load_report(rid), docs=docs, active="analysis")


@app.route("/parameters")
def parameters():
    reports = all_reports()
    rid = request.args.get("doc", "")
    selected = next((r for r in reports if r["id"] == rid), None)
    scope = [selected] if selected else reports
    agg = aggregate(scope)
    sentences = {l: [] for l in get_classifier().labels}
    for r in scope:
        for p in r["predictions"]:
            if not p["is_fragment"]:
                sentences[p["label"]].append({**p, "doc": r["file_name"], "doc_id": r["id"]})
    for items in sentences.values():
        items.sort(key=lambda p: p["decision_score"], reverse=True)
    return render_template("parameters.html", active="parameters", agg=agg, docs=reports,
                           selected=selected, sentences=sentences)


@app.route("/analytics")
def analytics():
    reports = all_reports()
    return render_template("analytics.html", active="analytics", docs=reports,
                           agg=aggregate(reports))


@app.route("/model")
def model():
    return render_template("model.html", ev=evaluation(), active="model")


@app.route("/settings")
def settings():
    import platform

    import fitz
    from importlib import metadata
    import sklearn
    info = {
        "Python": platform.python_version(),
        "scikit-learn (runtime)": sklearn.__version__,
        "scikit-learn (model saved with)": get_classifier().describe()["sklearn_version_saved"],
        "PyMuPDF": fitz.VersionBind,
        "Flask": metadata.version("flask"),
        "Data folder": DATA_DIR,
        "Reports stored": len(all_reports()),
        "Upload size limit": f"{MAX_UPLOAD_MB} MB (PDF only)",
        "Fragment threshold": f"≤ {FRAGMENT_MAX_WORDS} words",
    }
    return render_template("settings.html", active="settings", info=info)


# ------------------------------------------------------------------ actions/exports
@app.route("/analyze", methods=["POST"])
def analyze():
    if request.form.get("use_sample") == "1":
        report = run_analysis(SAMPLE_PDF_PATH, "SRS.pdf (bundled sample)")
        return redirect(url_for("report", rid=report["id"]))

    file = request.files.get("document")
    back = request.referrer or url_for("upload")
    if not file or not file.filename:
        flash("Please choose a PDF file to analyse.")
        return redirect(back)
    name = secure_filename(file.filename) or "document.pdf"
    if not name.lower().endswith(".pdf"):
        flash("Only PDF documents are supported (the project uses PyMuPDF for extraction).")
        return redirect(back)
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:12]}_{name}")
    file.save(path)
    try:
        report = run_analysis(path, file.filename)
    except Exception as exc:  # corrupt / encrypted PDFs
        flash(f"Could not analyse this PDF: {exc}")
        return redirect(back)
    if report["total_sentences"] == 0:
        flash("No text could be extracted from this PDF (it may be a scanned image).")
    return redirect(url_for("report", rid=report["id"]))


@app.route("/report/<rid>/download.csv")
def report_csv(rid):
    r = _load_report(rid)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["#", "sentence", "predicted_parameter", "svm_decision_score",
                     "word_count", "is_fragment"])
    for p in r["predictions"]:
        writer.writerow([p["index"], " ".join(p["sentence"].split()), p["label"],
                         p["decision_score"], p["word_count"], p["is_fragment"]])
    return Response(buf.getvalue(), mimetype="text/csv", headers={
        "Content-Disposition": f"attachment; filename=srs_analysis_{rid}.csv"})


@app.route("/report/<rid>/download.json")
def report_json(rid):
    r = _load_report(rid)
    return Response(json.dumps(r, indent=2, ensure_ascii=False), mimetype="application/json",
                    headers={"Content-Disposition": f"attachment; filename=srs_analysis_{rid}.json"})


@app.route("/reports/summary.csv")
def summary_csv():
    labels = get_classifier().labels
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["report_id", "document", "analysed", "pages", "sentences",
                     "requirement_sentences", "parameters_found"] + labels)
    for r in all_reports():
        writer.writerow([r["id"], r["file_name"], r["created"], r["pages"], r["total_sentences"],
                         r["requirement_sentences"], r["parameters_found"]]
                        + [r["label_counts"].get(l, 0) for l in labels])
    return Response(buf.getvalue(), mimetype="text/csv", headers={
        "Content-Disposition": "attachment; filename=srs_analysis_summary.csv"})


@app.route("/health")
def health():
    info = get_classifier().describe()
    return {"status": "ok", "model_steps": info["steps"], "labels": info["labels"]}


@app.errorhandler(413)
def too_large(_):
    flash(f"File is larger than the {MAX_UPLOAD_MB} MB limit.")
    return redirect(url_for("upload"))


@app.errorhandler(404)
def not_found(_):
    return render_template("empty.html", active="", title="Not found", icon="circle-alert",
                           message="That page or report does not exist."), 404


if __name__ == "__main__":
    host = os.environ.get("SRS_HOST", "127.0.0.1")
    port = int(os.environ.get("SRS_PORT", "5000"))
    get_classifier()  # load model once at startup
    print(f" * SRS Intelligence dashboard: http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
