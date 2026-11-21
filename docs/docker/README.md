# Docker deployment

A self-contained image packages the API, pinned Timestral base weights, LoRA adapter and calibration.

## Build and run

docker build -t retailmind-ai-forecasting:local .   |   docker run --rm -p 8000:8000 retailmind-ai-forecasting:local

## GPU and memory

Add --gpus all when NVIDIA Container Toolkit is available. The application sets a 2,048 MiB PyTorch GPU allocator cap; it can fall back to CPU.

## Verification

The corrected Dockerfile-built image passed scripts/ops/smoke_test_container.py locally. The image was approximately 13.7 GB. Docker build can require substantial temporary disk space.

## Model license

Review models/timestral209m_lora/LICENSE before distributing the model or image. The license includes a monthly revenue condition.

[Detailed reference](../milestone6-mlops.md) · [Visual guide](../../reports/docker.html) · [Project README](../../README.md)

## Docker Hub

[RetailMind AI Forecasting on Docker Hub](https://hub.docker.com/r/bmaged23/retailmind-ai-forecasting). The verified image is publicly available. Pull it with:

```sh
docker pull bmaged23/retailmind-ai-forecasting:latest
```
