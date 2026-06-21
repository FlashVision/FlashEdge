# FlashEdge Docker Image
# Build: docker build -t flashedge -f docker/Dockerfile .
# Run:   docker run --gpus all -it flashedge

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="FlashVision <flashvision@github.com>"
LABEL description="FlashEdge: Edge AI & TinyML toolkit"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY flashedge/ ./flashedge/

RUN pip install --no-cache-dir -e ".[all]"

COPY configs/ ./configs/
COPY examples/ ./examples/

ENTRYPOINT ["flashedge"]
CMD ["--help"]
