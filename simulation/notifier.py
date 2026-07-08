"""
Surge alert notifications. Uses real SMTP if all required variables are set in
.env / the environment; otherwise falls back to console-only output so the
simulation never breaks just because email isn't configured (the default for
this academic project - see README for how to enable real email).
"""
import os
import smtplib
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv()

REQUIRED_SMTP_VARS = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "ALERT_RECIPIENT"]


def _smtp_configured():
    return all(os.environ.get(v) for v in REQUIRED_SMTP_VARS)


def notify_surge(itemid, event_date, confidence, old_price, new_price, related_changes):
    header = (
        f"[SURGE ALERT] item {itemid} on {event_date}  "
        f"confidence={confidence:.2%}  price {old_price:.2f} -> {new_price:.2f}"
    )
    print(header)

    if related_changes:
        body_lines = [header, "Related items adjusted:"]
        for r in related_changes:
            body_lines.append(
                f"  - item {r['itemid']}: {r['old_price']:.2f} -> {r['new_price']:.2f} "
                f"({r['pct_change']:+.2f}%, co-occurrence weight={r['weight']})"
            )
            print(f"  -> related item {r['itemid']}: {r['old_price']:.2f} -> {r['new_price']:.2f} ({r['pct_change']:+.2f}%)")
    else:
        body_lines = [header, "No related items were bumped."]

    if not _smtp_configured():
        return

    try:
        msg = MIMEText("\n".join(body_lines))
        msg["Subject"] = f"SmartRetail surge alert - item {itemid}"
        msg["From"] = os.environ["SMTP_USER"]
        msg["To"] = os.environ["ALERT_RECIPIENT"]

        with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ["SMTP_PORT"])) as server:
            server.starttls()
            server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
            server.send_message(msg)
        print(f"  (email sent to {os.environ['ALERT_RECIPIENT']})")
    except Exception as e:
        print(f"  (email send failed, continuing without it: {e})")
