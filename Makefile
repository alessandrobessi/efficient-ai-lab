.PHONY: setup week1 week1-analyze clean

setup:
	uv sync

week1:
	uv run python experiments/01-inference-basics/scripts/exp_1_1_loading_time.py
	uv run python experiments/01-inference-basics/scripts/exp_1_2_prompt_length.py
	uv run python experiments/01-inference-basics/scripts/exp_1_3_output_length.py

week1-analyze:
	uv run python experiments/01-inference-basics/analysis/analyze.py

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
