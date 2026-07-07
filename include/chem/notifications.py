"""MS Teams notifications via an incoming webhook stored in an Airflow Variable."""
from __future__ import annotations
import logging
import requests
from airflow.models import Variable

log = logging.getLogger(__name__)


def _webhook_url() -> str | None:
    return Variable.get("msteams_webhook_url", default_var=None)


def send(title: str, text: str, color: str = "0076D7") -> None:
    url = _webhook_url()
    if not url:
        log.warning("msteams_webhook_url not set; skipping notification")
        return
    card = {
        "@type": "MessageCard", "@context": "http://schema.org/extensions",
        "themeColor": color, "summary": title,
        "sections": [{"activityTitle": title, "text": text}],
    }
    try:
        requests.post(url, json=card, timeout=10).raise_for_status()
    except Exception as e:                       # never fail the DAG on a notify error
        log.warning("Teams notification failed: %s", e)


def notify_failure(context) -> None:
    """on_failure_callback — fires for any failed task."""
    ti = context["task_instance"]
    send(f"Chem pipeline failed: {ti.dag_id}.{ti.task_id}",
         f"Run: {context['run_id']}\nLog: {ti.log_url}", color="D7263D")


def notify_success(dag_id: str, processed: list[str]) -> None:
    send(f"Chem pipeline succeeded: {dag_id}",
         f"Processed datasets: {', '.join(processed) if processed else 'none'}",
         color="2EB67D")