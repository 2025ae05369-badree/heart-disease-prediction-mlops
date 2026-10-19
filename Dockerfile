FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install --system -r requirements.txt

COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["python", "app.py"]
