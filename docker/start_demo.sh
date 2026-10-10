#!/bin/sh
# Start the API in the background, wait until it answers, then run the UI in the foreground.
set -e
uvicorn autoinsight.api:app --host 127.0.0.1 --port 8000 --log-level warning &
for i in $(seq 1 40); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" 2>/dev/null; then break; fi
  sleep 0.5
done
exec streamlit run app/streamlit_app.py --server.port="${PORT:-7860}" --server.address=0.0.0.0 \
  --browser.gatherUsageStats=false --server.enableXsrfProtection=false
