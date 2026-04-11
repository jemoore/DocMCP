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
RUN mkdir -p /data/docs /data/index

EXPOSE 8808

CMD ["uv", "run", "python", "-m", "docmcp.server"]
