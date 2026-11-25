"""Project tests. Run from the project root:  python -m unittest discover -s tests -v"""
import io
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dashboard"))

import tempfile  # noqa: E402

# Keep test uploads/reports out of the real dashboard folders.
os.environ["SRS_DATA_DIR"] = tempfile.mkdtemp(prefix="srs_test_")

import srs_classifier as sc  # noqa: E402


def notebook_predictions():
    """(sentence, label) pairs printed by the original Colab run of Final.ipynb."""
    with open(os.path.join(ROOT, "Final.ipynb"), encoding="utf-8") as f:
        nb = json.load(f)
    text = ""
    for out in nb["cells"][1]["outputs"]:
        if "text" in out:
            text += "".join(out["text"])
    pairs = []
    for block in text.split("Sentence: ")[1:]:
        sentence, _, rest = block.partition("\nPredicted Label: ")
        pairs.append((sentence, rest.strip().splitlines()[0]))
    return pairs


class TestFilesAndDependencies(unittest.TestCase):
    def test_required_files_exist(self):
        for p in (sc.PIPELINE_PATH, sc.LABEL_ENCODER_PATH, sc.DATASET_PATH, sc.SAMPLE_PDF_PATH):
            self.assertTrue(os.path.exists(p), p)

    def test_imports(self):
        import fitz, flask, joblib, pandas, sklearn  # noqa: F401
        self.assertEqual(sklearn.__version__, "1.5.2")

    def test_saved_version_matches_runtime(self):
        import sklearn
        self.assertEqual(sc.saved_sklearn_version(), sklearn.__version__)

    def test_dataset(self):
        import pandas as pd
        data = pd.read_csv(sc.DATASET_PATH)
        self.assertEqual(list(data.columns), ["Sentence", "Label"])
        self.assertEqual(len(data.dropna()), 94)
        self.assertEqual(data["Label"].dropna().nunique(), 8)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = sc.get_classifier()

    def test_pipeline_structure(self):
        steps = [cls for _, cls in self.clf.describe()["steps"]]
        self.assertEqual(steps, ["TfidfVectorizer", "LinearSVC"])
        self.assertEqual(len(self.clf.labels), 8)

    def test_reproduces_notebook_accuracy(self):
        from evaluate_model import evaluate
        self.assertAlmostEqual(evaluate()["accuracy"] * 100, 78.95, places=2)

    def test_preprocess_text(self):
        s = sc.preprocess_text("Uptime is 99.99%. Is backup daily? Yes e.g. nightly.")
        self.assertEqual(s, ["Uptime is 99.99%.", "Is backup daily?", "Yes e.g. nightly."])

    def test_training_sentences_classified(self):
        r = self.clf.classify(["A 99.99% uptime guarantee ensures continuous service."])[0]
        self.assertEqual(r.label, "Availability")
        self.assertIsInstance(r.decision_score, float)

    def test_pdf_matches_original_notebook_output(self):
        """Every sentence + label printed in the Colab notebook is reproduced locally."""
        expected = notebook_predictions()
        self.assertGreater(len(expected), 30)
        got = self.clf.analyze_pdf(sc.SAMPLE_PDF_PATH)
        self.assertEqual(len(got), len(expected))
        for (exp_sentence, exp_label), pred in zip(expected, got):
            # Colab's console replaced non-ASCII characters with U+FFFD; compare labels + ASCII text.
            norm = lambda t: "".join(ch for ch in t if ord(ch) < 128).split()
            self.assertEqual(norm(pred.sentence), norm(exp_sentence))
            self.assertEqual(pred.label, exp_label, pred.sentence)


class TestDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as dashboard_app
        dashboard_app.app.config["TESTING"] = True
        cls.client = dashboard_app.app.test_client()

    def test_pages(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/health").json["status"], "ok")
        model = self.client.get("/model")
        self.assertEqual(model.status_code, 200)
        self.assertIn(b"78.95%", model.data)

    def test_upload_pdf_end_to_end(self):
        with open(sc.SAMPLE_PDF_PATH, "rb") as f:
            resp = self.client.post("/analyze", data={"document": (io.BytesIO(f.read()), "SRS.pdf")},
                                    content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 302)
        report_url = resp.headers["Location"]
        page = self.client.get(report_url)
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Identified parameters", page.data)
        rid = report_url.rstrip("/").split("/")[-1]
        csv_resp = self.client.get(f"/report/{rid}/download.csv")
        self.assertIn("predicted_parameter", csv_resp.get_data(as_text=True).splitlines()[0])
        data = self.client.get(f"/report/{rid}/download.json").get_json()
        self.assertEqual(data["total_sentences"], len(notebook_predictions()))

    def test_all_sidebar_pages_render(self):
        import app as dashboard_app
        for key, label, _icon, endpoint in dashboard_app.NAV:
            with dashboard_app.app.test_request_context():
                url = dashboard_app.url_for(endpoint)
            resp = self.client.get(url, follow_redirects=True)
            self.assertEqual(resp.status_code, 200, url)
            self.assertIn(b'class="sb-link active"', resp.data, f"{url} has no active nav item")
            self.assertIn(b"lucide.min.js", resp.data)

    def test_dashboard_uses_real_data(self):
        with open(sc.SAMPLE_PDF_PATH, "rb") as f:
            self.client.post("/analyze", data={"document": (io.BytesIO(f.read()), "SRS.pdf")},
                             content_type="multipart/form-data")
        import app as dashboard_app
        agg = dashboard_app.aggregate(dashboard_app.all_reports())
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("78.95%", html)
        self.assertIn(f'<div class="v">{agg["documents"]}</div>', html)
        self.assertIn(f'<div class="v">{agg["sentences"]}</div>', html)
        self.assertIn("Parameter distribution", html)
        for page in ("/analytics", "/parameters", "/documents", "/documents?q=srs"):
            self.assertIn(b"SRS.pdf", self.client.get(page).data, page)

    def test_summary_csv(self):
        text = self.client.get("/reports/summary.csv").get_data(as_text=True)
        self.assertTrue(text.startswith("report_id,document,analysed"))

    def test_analysis_redirects_to_latest_report(self):
        resp = self.client.get("/analysis")
        self.assertIn(resp.status_code, (200, 302))

    def test_rejects_non_pdf(self):
        resp = self.client.post("/analyze", data={"document": (io.BytesIO(b"hello"), "notes.txt")},
                                content_type="multipart/form-data", follow_redirects=True)
        self.assertIn(b"Only PDF documents are supported", resp.data)

    def test_unknown_report_404(self):
        self.assertEqual(self.client.get("/report/doesnotexist").status_code, 404)


if __name__ == "__main__":
    unittest.main()
