#!/bin/bash

# Navigate to backend and start FastAPI in the background (NO reload in production!)
cd /app/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Wait for the backend to start
sleep 5

# Navigate to frontend and start Streamlit
cd /app/frontend
streamlit run app.py --server.port=7860 --server.address=0.0.0.0
