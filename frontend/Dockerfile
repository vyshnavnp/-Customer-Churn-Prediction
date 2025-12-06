FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the dashboard code
COPY dashboard.py .

# We don't copy data here; we will mount it via Docker Compose
# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]