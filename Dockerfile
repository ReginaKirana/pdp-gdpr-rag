FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# dependencies first, so this layer is cached until pyproject.toml / uv.lock change
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

COPY src ./src
COPY eval ./eval
COPY app.py ./

EXPOSE 7860
CMD ["python", "app.py"]
