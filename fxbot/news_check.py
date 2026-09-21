"""Read-only calendar smoke check; never connects to MT5 or submits orders."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv

from fxbot.config import settings_from_env
from fxbot.news import build_news_gateway


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="Overlay this profile on the existing environment for this check only")
    args = parser.parse_args(argv)
    if args.env_file is not None:
        if not args.env_file.is_file():
            parser.error("--env-file must refer to an existing file")
        load_dotenv(args.env_file, override=True)
    settings = settings_from_env()
    now = datetime.now(timezone.utc)
    # A probe must not change the running worker's persistent news state.
    with TemporaryDirectory(prefix="fx-news-check-") as directory:
        gateway = build_news_gateway(
            settings.strategy,
            database_url=f"sqlite:///{Path(directory) / 'news.db'}",
        )
        snapshot = gateway.ensure_current(now)
    upcoming = sorted(
        (event for event in snapshot.events if event.starts_at >= now and event.impact_score >= settings.strategy.news_blackout_impact_score_min),
        key=lambda event: event.starts_at,
    )
    print(json.dumps({
        "source": snapshot.source,
        "state": snapshot.known_state,
        "event_count": len(snapshot.events),
        "last_error": snapshot.last_error,
        "next_high_impact": [
            {"name": event.name, "currency": event.currency, "starts_at": event.starts_at.isoformat(), "impact_score": event.impact_score}
            for event in upcoming[:5]
        ],
    }, indent=2))
    return 1 if snapshot.stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
