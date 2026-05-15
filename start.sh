#!/bin/bash

# Navigate to backend and start FastAPI in the background
cd backend
python -m app.main &

# Wait a few seconds to let the backend start
sleep 5

# Navigate to frontend and start Streamlit
cd ../frontend
# We need to tell Streamlit to run on 7860 and bind to 0.0.0.0 for Docker/HuggingFace
streamlit run app.py --server.port=7860 --server.address=0.0.0.0
