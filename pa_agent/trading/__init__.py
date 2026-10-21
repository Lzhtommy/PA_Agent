"""Automated execution of stage-2 decisions on OKX perpetual swaps.

Flow::

    GUI / headless runner --emit_signal()--> SQLite (signals)
    trading service (python -m pa_agent.trading) polls signals
        -> risk checks -> OKX orders -> fills / positions / PnL -> SQLite
        -> dashboard (FastAPI) reads SQLite
"""
