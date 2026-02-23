#!/bin/bash
# Source .env so Streamlit sees CHROMA_DIR, GROQ_API_KEY, etc. when run by systemd.
cd /opt/Epstinefiles
set -a
[ -f .env ] && . ./.env
set +a
exec /opt/Epstinefiles/.venv/bin/streamlit run search_ui.py --server.port 8501 --server.address 0.0.0.0
