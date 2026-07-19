"""
send_email() — sends the final RAG answer as an email body when advanced_filter
mode detects an email-send request in the query (see email_trigger.py, api.py).

Credentials are read from environment variables (rag_pipeline/.env), never
hardcoded — set these two vars there (lowercase, matching this repo's existing
groq_api / deep_gram_key convention):
  email_sender_address       - the Gmail address to send from
  email_sender_app_password  - a Gmail App Password (NOT the normal account
                                password; generate one at myaccount.google.com/apppasswords)
"""
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv()


def send_email(receiver_email: str, subject: str, body: str) -> None:
    """Send `body` as a plaintext email to `receiver_email` via Gmail SMTP.

    Raises on any failure (missing credentials, SMTP/auth error, network
    error) — callers that must not let a failure here affect their own
    response (see api.py) are responsible for catching it themselves.
    """
    sender_email = os.getenv("email_sender_address")
    sender_password = os.getenv("email_sender_app_password")
    if not sender_email or not sender_password:
        raise RuntimeError("email_sender_address / email_sender_app_password not set in environment")

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
