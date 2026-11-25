"""Local, reusable version of the SRS classification logic in final.py / Final.ipynb.

The original code only runs in Google Colab (``google.colab.files.upload`` and
``!pip install``). This module keeps the original method unchanged:

* PDF text extraction with PyMuPDF (``page.get_text()`` for every page)
* the same regular expression for sentence splitting (``preprocess_text``)
* the saved TF-IDF + LinearSVC pipeline (``sentence_classifier_pipeline.pkl``)
* the saved LabelEncoder (``label_encoder.pkl``) to turn class ids into names

Nothing is retrained here. The only addition is that the raw LinearSVC
``decision_function`` margin of the predicted class is returned. It is
a real model output, but it is NOT a probability.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from typing import List, Optional

import joblib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(PROJECT_ROOT, "sentence_classifier_pipeline.pkl")
LABEL_ENCODER_PATH = os.path.join(PROJECT_ROOT, "label_encoder.pkl")
DATASET_PATH = os.path.join(PROJECT_ROOT, "Dataset.csv")
SAMPLE_PDF_PATH = os.path.join(PROJECT_ROOT, "SRS.pdf")

# Sentences with fewer words than this are usually headings or numbering
# ("2.", "3.1 Establishment") left over by the original regex splitter.
FRAGMENT_MAX_WORDS = 3


def preprocess_text(text: str) -> List[str]:
    """Split text into sentences. Copied verbatim from final.py."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from every page of a PDF, as in final.py."""
    import fitz  # PyMuPDF

    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text


def pdf_page_count(pdf_path: str) -> int:
    import fitz

    with fitz.open(pdf_path) as pdf:
        return pdf.page_count


def saved_sklearn_version(path: str = PIPELINE_PATH) -> str:
    """scikit-learn version recorded inside the pickle (sklearn drops it on load)."""
    with open(path, "rb") as f:
        m = re.search(rb"_sklearn_version\x94\x8c.(\d+\.\d+\.\d+)", f.read())
    return m.group(1).decode() if m else "unknown"


@dataclass
class Prediction:
    index: int
    sentence: str
    label: str
    decision_score: float
    word_count: int
    is_fragment: bool

    def to_dict(self):
        return asdict(self)


class SRSClassifier:
    """Wraps the saved pipeline and label encoder."""

    def __init__(self, pipeline_path: str = PIPELINE_PATH,
                 label_encoder_path: str = LABEL_ENCODER_PATH):
        self.pipeline = joblib.load(pipeline_path)
        self.label_encoder = joblib.load(label_encoder_path)

    @property
    def labels(self) -> List[str]:
        return [str(c) for c in self.label_encoder.classes_]

    def describe(self) -> dict:
        steps = [(name, type(step).__name__) for name, step in self.pipeline.steps]
        tfidf = self.pipeline.steps[0][1]
        return {
            "steps": steps,
            "vocabulary_size": len(getattr(tfidf, "vocabulary_", {})),
            "labels": self.labels,
            "sklearn_version_saved": saved_sklearn_version(),
        }

    def classify(self, sentences: List[str]) -> List[Prediction]:
        if not sentences:
            return []
        codes = self.pipeline.predict(sentences)
        names = self.label_encoder.inverse_transform(codes)
        scores = self.pipeline.decision_function(sentences)
        classes = list(self.pipeline.classes_)
        results = []
        for i, (sentence, code, name) in enumerate(zip(sentences, codes, names)):
            score = float(scores[i][classes.index(code)])
            words = len(sentence.split())
            results.append(Prediction(
                index=i + 1,
                sentence=sentence,
                label=str(name),
                decision_score=round(score, 4),
                word_count=words,
                is_fragment=words <= FRAGMENT_MAX_WORDS,
            ))
        return results

    def analyze_text(self, text: str) -> List[Prediction]:
        return self.classify(preprocess_text(text))

    def analyze_pdf(self, pdf_path: str) -> List[Prediction]:
        return self.analyze_text(extract_pdf_text(pdf_path))


_default: Optional[SRSClassifier] = None


def get_classifier() -> SRSClassifier:
    global _default
    if _default is None:
        _default = SRSClassifier()
    return _default
