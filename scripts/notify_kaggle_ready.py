#!/usr/bin/env python3
"""Send email when a solve is pushed and ready for manual Kaggle upload."""

from __future__ import annotations

import argparse
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path


def build_subject(package_path: Path) -> str:
    # kaggle-submissions/2026-06-19/submission-1/kaggle_submit_package.md
    parts = package_path.parts
    try:
        idx = parts.index("kaggle-submissions")
        date = parts[idx + 1]
        sub = parts[idx + 2].replace("submission-", "")
        return f"Kaggle ready: {date} submission-{sub}"
    except (ValueError, IndexError):
        return "Kaggle ready: NeuroGolf submission package"


def send_email(subject: str, body: str) -> bool:
    to_addr = os.environ.get("NOTIFY_EMAIL", "ilakkmanoharan@gmail.com")
    smtp_user = os.environ.get("SMTP_USER", to_addr)
    smtp_password = os.environ.get("SMTP_PASSWORD") or os.environ.get("GMAIL_APP_PASSWORD")
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))

    if not smtp_password:
        print("SMTP_PASSWORD (or GMAIL_APP_PASSWORD) not set — skip email send")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_addr
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls(context=context)
        server.login(smtp_user, smtp_password)
        server.send_message(msg)

    print(f"Email sent to {to_addr}: {subject}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_md", type=Path, help="Path to kaggle_submit_package.md")
    args = parser.parse_args()

    package = args.package_md
    if not package.is_file():
        print(f"Missing package file: {package}")
        return 1

    body = package.read_text(encoding="utf-8")
    subject = build_subject(package)
    repo = os.environ.get("GITHUB_REPOSITORY", "ilakkmanoharan/ARC-NeuroGolf")
    body = (
        f"{body}\n\n"
        f"---\n"
        f"Repo: https://github.com/{repo}\n"
        f"Folder: {package.parent.as_posix()}/\n"
    )

    sent = send_email(subject, body)
    print(f"Subject: {subject}")
    return 0 if sent or not os.environ.get("SMTP_PASSWORD") else 1


if __name__ == "__main__":
    raise SystemExit(main())
