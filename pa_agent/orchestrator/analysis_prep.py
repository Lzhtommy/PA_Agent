"""Frame building + incremental-base lookup before an analysis (Qt-free)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnalysisPrepResult:
    frame: Any
    previous_record: Any | None
    incremental_new_bar_count: int | None
    incremental_detail: str | None


def prepare_analysis(
    *,
    bars_raw: list[Any],
    symbol: str,
    timeframe: str,
    bar_count: int,
    now_ms: int,
    force_incremental: bool,
    incremental_threshold: int,
) -> AnalysisPrepResult | None:
    """Build the KlineFrame and find the previous record for incremental analysis.

    Returns None when there are not enough bars to build a frame.
    """
    from pa_agent.data.snapshot import build_display_frame
    from pa_agent.records.analysis_history import (
        compute_incremental_bar_delta,
        find_latest_successful_record,
        format_bar_ts,
    )

    frame = build_display_frame(list(bars_raw), bar_count, symbol, timeframe, now_ms=now_ms)
    if frame is None:
        return None

    previous = None
    incremental_new_bar_count: int | None = None
    incremental_detail: str | None = None

    if force_incremental or incremental_threshold > 0:
        previous = find_latest_successful_record(symbol=symbol, timeframe=timeframe)
        if previous is not None:
            delta = compute_incremental_bar_delta(frame, previous)
            if delta is not None:
                new_count = delta.new_count
                if force_incremental or new_count <= incremental_threshold:
                    incremental_new_bar_count = new_count
                    anchor_label = format_bar_ts(delta.anchor_ts_open)
                    if new_count == 0:
                        incremental_detail = f"锚定K线 {anchor_label}，无新增已收盘K线"
                    elif new_count == 1:
                        incremental_detail = (
                            f"锚定K线 {anchor_label}，新增1根 "
                            f"{format_bar_ts(delta.new_bar_ts_opens[0])}"
                        )
                    else:
                        newest = format_bar_ts(delta.new_bar_ts_opens[0])
                        oldest_new = format_bar_ts(delta.new_bar_ts_opens[-1])
                        incremental_detail = (
                            f"锚定K线 {anchor_label}，新增{new_count}根"
                            f"（{oldest_new} → {newest}）"
                        )
                elif not force_incremental:
                    previous = None

    return AnalysisPrepResult(
        frame=frame,
        previous_record=previous,
        incremental_new_bar_count=incremental_new_bar_count,
        incremental_detail=incremental_detail,
    )
