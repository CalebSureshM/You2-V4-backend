FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything in the agent directory (including .env, .py files, users.json, etc.)
COPY . .

CMD ["uvicorn", "fastapi_backend:app", "--host", "0.0.0.0", "--port", "8080"]
