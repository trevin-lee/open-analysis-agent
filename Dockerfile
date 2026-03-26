FROM python:3.12-slim

ARG VERSION=dev

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    voms-clients-java \
    && rm -rf /var/lib/apt/lists/*

# Install package
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install --no-cache-dir -e ".[full]"

# Set version label
LABEL org.opencontainers.image.title="Open Analysis Agent" \
      org.opencontainers.image.description="AI-powered research assistant for CMS experiment data analysis at the LHC" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/trevin-lee/OpenAnalysisAgent" \
      org.opencontainers.image.licenses="MIT"

ENV OAA_VERSION=${VERSION}

# Workspace for user data
RUN mkdir -p /workspace/data /workspace/output /workspace/artifacts
WORKDIR /workspace

ENTRYPOINT ["python", "-m", "src.terminal"]
