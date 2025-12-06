# Start from a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# --- ADDED SECTION ---
# Install libgomp1 which is required for CatBoost on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# ---------------------

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and the trained model
COPY main.py .
COPY schemas.py .
COPY final_model.cbm .

# Expose the port the app runs on
EXPOSE 80

# Command to run the API when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]