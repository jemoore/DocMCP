FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock* ./

# Install production dependencies
RUN uv sync --no-dev --no-install-project

# Copy source code
COPY src/ src/

# Install the project itself
RUN uv sync --no-dev

# Create data directories
RUN mkdir -p /data/docs /data/index /data/hf-cache

# Keep the sentence-transformers model download in a mountable location so it
# survives container recreation (see the docmcp-hf-cache volume in compose).
ENV HF_HOME=/data/hf-cache

EXPOSE 8808

CMD ["uv", "run", "python", "-m", "docmcp.server"]
