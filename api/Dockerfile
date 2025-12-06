# Start from a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

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
# Note: We bind to 0.0.0.0 for external access
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]