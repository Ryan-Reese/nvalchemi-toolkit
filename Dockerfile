# uses a lightweight Ubuntu-based image with CUDA 13
FROM nvidia/cuda:13.2.0-runtime-ubuntu24.04

# grab package updates and other system dependencies here
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    g++ \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*
RUN git config --global --add safe.directory /nvalchemi-toolkit
# copy uv for venv management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv venv --seed --python 3.12 /opt/venv
# this sets the default virtual environment to use
ENV VIRTUAL_ENV=/opt/venv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# install ALCHEMI Toolkit-Ops
WORKDIR /nvalchemi-toolkit
COPY . .
RUN uv sync --all-extras --group docs
RUN uv pip install torch --reinstall --index-url https://download.pytorch.org/whl/cu130
RUN uv pip install jupyterlab jupyterlab-vim

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
