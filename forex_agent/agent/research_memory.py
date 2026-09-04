from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from forex_agent.data.schemas import ResearchHypothesis, HypothesisStatus
from forex_agent.log import get_logger

logger = get_logger(__name__)


class ResearchMemory:
    """Persistent storage for findings, hypotheses, experiments, and decisions.

    Stores data in a JSONL file so it survives between agent runs.
    """

    def __init__(self, path: str = "data/research_memory.jsonl") -> None:
        self.path = Path(path)
        self._hypotheses: list[ResearchHypothesis] = []
        self._findings: list[dict[str, Any]] = []
        self._decisions: list[dict[str, Any]] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rtype = record.get("type")
                    if rtype == "hypothesis":
                        self._hypotheses.append(ResearchHypothesis.from_dict(record.get("data", {})))
                    elif rtype == "finding":
                        self._findings.append(record.get("data", {}))
                    elif rtype == "decision":
                        self._decisions.append(record.get("data", {}))
        except Exception as exc:
            logger.warning("Failed to load research memory: %s", exc)

    def _append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # --- Hypotheses ---

    def add_hypothesis(self, hypothesis: ResearchHypothesis) -> None:
        self._ensure_loaded()
        if not hypothesis.date_created:
            hypothesis.date_created = datetime.utcnow().isoformat()
        self._hypotheses.append(hypothesis)
        self._append({"type": "hypothesis", "data": hypothesis.to_dict()})

    def update_hypothesis_status(
        self, hypothesis_text: str, new_status: HypothesisStatus, result: str = ""
    ) -> bool:
        self._ensure_loaded()
        for h in self._hypotheses:
            if h.hypothesis == hypothesis_text:
                h.status = new_status
                if result:
                    h.result = result
                self._append({"type": "hypothesis", "data": h.to_dict()})
                return True
        return False

    def get_hypotheses(self, status: HypothesisStatus | None = None) -> list[ResearchHypothesis]:
        self._ensure_loaded()
        if status is None:
            return list(self._hypotheses)
        return [h for h in self._hypotheses if h.status == status]

    # --- Findings ---

    def add_finding(self, finding: dict[str, Any]) -> None:
        self._ensure_loaded()
        finding.setdefault("date", datetime.utcnow().isoformat())
        self._findings.append(finding)
        self._append({"type": "finding", "data": finding})

    def get_findings(self) -> list[dict[str, Any]]:
        self._ensure_loaded()
        return list(self._findings)

    # --- Decisions ---

    def add_decision(self, decision: dict[str, Any]) -> None:
        self._ensure_loaded()
        decision.setdefault("date", datetime.utcnow().isoformat())
        self._decisions.append(decision)
        self._append({"type": "decision", "data": decision})

    def get_decisions(self) -> list[dict[str, Any]]:
        self._ensure_loaded()
        return list(self._decisions)

    # --- Summary ---

    def summary(self) -> dict[str, Any]:
        self._ensure_loaded()
        return {
            "total_hypotheses": len(self._hypotheses),
            "by_status": {
                status.value: sum(1 for h in self._hypotheses if h.status == status)
                for status in HypothesisStatus
            },
            "total_findings": len(self._findings),
            "total_decisions": len(self._decisions),
        }
