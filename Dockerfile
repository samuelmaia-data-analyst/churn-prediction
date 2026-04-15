FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements-runtime.txt requirements-runtime.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-runtime.txt

COPY . .

CMD ["python", "-m", "src.cli.pipeline", "--data-dir", "data", "--log-level", "INFO", "--environment", "dev"]
