.PHONY: run-backend run-frontend test

run-backend:
	uvicorn src.omnifind.api.main:app --reload

run-frontend:
	streamlit run src/omnifind/ui/streamlit_app.py

test:
	pytest -q
