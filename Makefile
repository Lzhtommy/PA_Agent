UV ?= uv

# ===== 原有指令（直接使用当前环境的 python / pytest / ruff）=====

# 启动 GUI
run:
	python -m pa_agent.main

# 运行测试
test:
	pytest -q

# 代码检查
lint:
	ruff check . && black --check .

# ===== 对应 uv 隔离环境版本 =====

# 自动创建隔离环境并安装依赖（含 dev 工具 + 交易/面板依赖）
.venv: pyproject.toml
	$(UV) sync --extra dev --extra trading

# 使用 uv 启动 GUI
uv-run: .venv
	$(UV) run python -m pa_agent.main

# 使用 uv 运行测试
uv-test: .venv
	$(UV) run pytest -q

# 使用 uv 代码检查
uv-lint: .venv
	$(UV) run ruff check . && $(UV) run black --check .

# ===== 量化交易（详见 docs/量化交易使用说明.md）=====

SYMBOLS ?= BTC-USDT-SWAP
TF ?= 1h

# 检查 OKX 连接、密钥和账户模式
trade-check: .venv
	$(UV) run python -m pa_agent.trading check

# 只跑交易执行器 + 收益面板（信号来自桌面程序的分析）
trade-run: .venv
	$(UV) run python -m pa_agent.trading run

# 无窗口：后台持续分析 + 交易执行器 + 收益面板（VPS 推荐）
headless: .venv
	$(UV) run python -m pa_agent.headless --symbols $(SYMBOLS) --timeframe $(TF) --trade

# 收益与 AI 质量报告
trade-report: .venv
	$(UV) run python -m pa_agent.trading report

# 紧急停止 / 解除
trade-kill: .venv
	$(UV) run python -m pa_agent.trading kill
trade-resume: .venv
	$(UV) run python -m pa_agent.trading resume

# 启用 pre-commit，防止 settings / 日志 / 记录被提交
setup-secrets:
	powershell -ExecutionPolicy Bypass -File tools/setup_git_secrets.ps1

.PHONY: run test lint uv-run uv-test uv-lint setup-secrets trade-check trade-run headless trade-report trade-kill trade-resume
