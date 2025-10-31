FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# Copy application code
COPY app.py ./app.py
COPY static_model/ ./static_model/
COPY OBD.db ./OBD.db

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host","0.0.0.0","--port","8001"]

