# -----------------------------
# 1. Base image
# -----------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -----------------------------
# 2. System deps
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# 3. Install Poetry
# I have matched this to my project Poetry version
# -----------------------------
ENV POETRY_VERSION=2.2.1
RUN curl -sSL https://install.python-poetry.org | python3 -

# Poetry will be installed into /root/.local/bin
ENV PATH="/root/.local/bin:$PATH"

# -----------------------------
# 4. Disable Poetry virtualenvs
# Dependencies installed directly into system Python of the container
# -----------------------------
RUN poetry config virtualenvs.create false

# -----------------------------
# 5. Set workdir
# -----------------------------
WORKDIR /app

# -----------------------------
# 6. Copy only dependency files first
# -----------------------------
COPY pyproject.toml poetry.lock* ./

# -----------------------------
# 7. Install dependencies
# -----------------------------
RUN poetry install --no-interaction --no-ansi

# -----------------------------
# 8. Copy project code
# -----------------------------
COPY epml_da ./epml_da
COPY models ./models

# I am not sure what to do with data, might need to add processed later
# COPY data/processed ./data/processed

# -----------------------------
# 9. Default command is calling model to predict
# -----------------------------
CMD ["python", "-m", "epml_da.modeling.predict"]
