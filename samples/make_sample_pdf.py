"""Create samples/sample_cloud_storage_srs.pdf, a small SRS written for this demo.

None of these sentences appear in Dataset.csv, so the dashboard's output for
this file shows how the saved model handles text it has not seen.
Run: python samples/make_sample_pdf.py
"""
import os

import fitz  # PyMuPDF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sample_cloud_storage_srs.pdf")

SECTIONS = [
    ("Software Requirements Specification - Campus Cloud Storage Service", None),
    ("1. Purpose", "This document lists the requirements a university uses to choose a cloud storage vendor "
     "for student and staff files."),
    ("2. Vendor Background", "The vendor shall have operated a public cloud business for at least ten years. "
     "The vendor should be listed among the top three providers by global market share."),
    ("3. Availability", "The service must guarantee 99.95% monthly uptime with automatic failover between regions. "
     "Planned maintenance must not cause any downtime for users."),
    ("4. Pricing", "The vendor shall offer a pay-as-you-go plan and discounted annual subscriptions. "
     "Academic institutions should receive reduced pricing tiers."),
    ("5. Data Migration", "The vendor must provide physical storage devices for offline transfer of large datasets. "
     "Bulk data shipped on encrypted drives should be imported within five working days."),
    ("6. Integration", "The service must integrate with Microsoft Office 365 and Active Directory single sign-on. "
     "Users should be able to open Word and Excel files directly from storage."),
    ("7. Backup and Recovery", "Daily backups shall be replicated to a second geographic region. "
     "The recovery time objective after a regional outage must be under four hours."),
    ("8. Storage Options", "The vendor must offer object storage, block storage and file shares. "
     "Archive storage tiers should be available for rarely accessed research data."),
]


def build():
    doc = fitz.open()
    page = doc.new_page()
    y = 60
    for heading, body in SECTIONS:
        size = 15 if body is None else 12
        page.insert_text((60, y), heading, fontsize=size, fontname="helv")
        y += 22
        if body:
            rect = fitz.Rect(60, y - 10, 540, y + 70)
            page.insert_textbox(rect, body, fontsize=10.5, fontname="helv")
            y += 72
    doc.save(OUT)
    print("Wrote", OUT)


if __name__ == "__main__":
    build()
