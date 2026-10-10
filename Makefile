.PHONY: install test lint eval run serve ui sandbox-image docker-up docker-up-llm samples clean

install:            ## install everything in editable mode
	pip install -c constraints.txt -e ".[all]"

test:               ## run the test suite (~5 min; includes full pipeline runs)
	pytest

lint:
	ruff check .

eval:               ## reproduce docs/EVALUATION.md
	python scripts/evaluate.py

run:                ## analyse the churn sample from the command line
	autoinsight run data/sample/telco_churn.csv -q "Find the driving factors behind customer churn"

serve:              ## REST API on :8000 (docs at /docs)
	autoinsight serve

ui:                 ## Streamlit UI on :8501 (start `make serve` first)
	streamlit run app/streamlit_app.py

sandbox-image:      ## build the hardened Docker sandbox used for LLM-written code
	docker build -f docker/Dockerfile.sandbox -t autoinsight-sandbox:latest .

docker-up:          ## API + UI in containers (offline mode)
	docker compose up --build

docker-up-llm:      ## API + UI + real LLM + Docker sandbox (needs API key in .env)
	docker compose -f docker-compose.yml -f docker-compose.llm.yml up --build

samples:            ## regenerate the synthetic sample datasets
	python data/generate_sample_data.py

clean:
	rm -rf runs .pytest_cache .ruff_cache
