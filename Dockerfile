# syntax=docker/dockerfile:1
# Station demo: the live 3D dashboard and its simulation API in one image.
#
#   docker build -t bus-rl-station .
#   docker run --rm -p 8000:8000 bus-rl-station      # open http://localhost:8000
#
# The runtime stage carries only what the station server imports (numpy, gymnasium,
# the FastAPI stack and the native kernel); torch and the training stack stay out.

FROM rust:1.97.1-slim-bookworm AS native
# PyO3 reads the interpreter config at build time; bookworm's python3 is 3.11 (abi3-py311).
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
# The cache mounts keep crate downloads and incremental builds across rebuilds;
# the .so is copied out because the target cache is not part of the layer.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release --locked -p bus-sim-python \
    && cp target/release/libbus_sim_native.so /build/bus_sim_native.so

FROM python:3.11-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /usr/local/bin/uv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy
# Versions match uv.lock. Only the server's runtime imports are installed, not the
# project's full dependency set (which pulls in torch).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    numpy==2.4.6 \
    gymnasium==1.3.0 \
    fastapi==0.143.5 \
    pydantic-settings==2.16.0 \
    uvicorn==0.54.0
WORKDIR /app
COPY configs ./configs
COPY src/bus_rl ./src/bus_rl
COPY src/bus_sim ./src/bus_sim
COPY web/station ./web/station
COPY --from=native /build/bus_sim_native.so ./src/bus_sim_native.so
# The scenario days are gitignored but deterministic: regenerate the held-out
# days the station replays instead of copying a local data/ folder.
RUN python -m bus_rl.cli generate --config configs/base.toml \
        --output data/generated/base --counts validation=2
RUN useradd --create-home --uid 10001 station
USER station
ENV SERVER__HOST=0.0.0.0 \
    SIMULATION__MANIFEST_PATH=data/generated/base/manifest.json
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "bus_rl.main:app", "--host", "0.0.0.0", "--port", "8000"]
