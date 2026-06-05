FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEMO_PUBLIC_MODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt

COPY demo_site demo_site
COPY media media
COPY scripts/serve_demo.py scripts/serve_demo.py
COPY scripts/process_upload_demo.py scripts/process_upload_demo.py
COPY PORTFOLIO.md README.md ./

EXPOSE 8080

CMD ["python", "scripts/serve_demo.py"]
