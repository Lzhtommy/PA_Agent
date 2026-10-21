"""Background preparation before starting the AI analysis worker."""
from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from pa_agent.orchestrator.analysis_prep import AnalysisPrepResult, prepare_analysis

logger = logging.getLogger(__name__)


class AnalysisPrepWorker(QThread):
    """Build KlineFrame + incremental base lookup off the UI thread."""

    ready = pyqtSignal(object)  # AnalysisPrepResult
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        bars_raw: list[Any],
        symbol: str,
        timeframe: str,
        bar_count: int,
        now_ms: int,
        force_incremental: bool,
        incremental_threshold: int,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._bars_raw = list(bars_raw)
        self._symbol = symbol
        self._timeframe = timeframe
        self._bar_count = bar_count
        self._now_ms = now_ms
        self._force_incremental = force_incremental
        self._incremental_threshold = incremental_threshold

    def run(self) -> None:
        try:
            result = prepare_analysis(
                bars_raw=self._bars_raw,
                symbol=self._symbol,
                timeframe=self._timeframe,
                bar_count=self._bar_count,
                now_ms=self._now_ms,
                force_incremental=self._force_incremental,
                incremental_threshold=self._incremental_threshold,
            )
            if result is None:
                self.failed.emit("数据不足，无法构建分析快照")
                return
            self.ready.emit(result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Analysis prep failed: %s", exc)
            self.failed.emit(str(exc))
