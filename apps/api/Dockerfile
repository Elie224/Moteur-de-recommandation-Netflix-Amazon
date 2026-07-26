FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY packages/recommendation_engine /app/packages/recommendation_engine
COPY apps/api /app/apps/api

RUN pip install --no-cache-dir /app/packages/recommendation_engine /app/apps/api

WORKDIR /app/apps/api
CMD ["sh", "-c", "alembic upgrade head && uvicorn recommender_api.main:app --host 0.0.0.0 --port 8000"]
