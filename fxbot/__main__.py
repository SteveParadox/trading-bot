"""Run the FX forward-test worker directly."""

from __future__ import annotations

import argparse
import asyncio
import logging

from fxbot.config import settings_from_env
from fxbot.forward import ForwardTestWorker
from fxbot.models import BotRunState


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MT5 FX forward-test worker")
    parser.add_argument("--start", action="store_true", help="set bot state to running before entering the loop")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )
    worker = ForwardTestWorker(settings_from_env())
    if args.start:
        worker.journal.set_state(BotRunState.RUNNING, "worker_cli_start")
    asyncio.run(worker.run_forever())


if __name__ == "__main__":
    main()
