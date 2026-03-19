run:
	uv run bot

fmt:
	uvx ruff format .

lint:
	uvx ruff check .

test:
	uv run pytest

sync:
	uv sync
