FROM python:3.11-slim-bookworm

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.lock pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.lock

COPY README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 8001
CMD ["retail", "api", "--host", "0.0.0.0", "--port", "8001"]
