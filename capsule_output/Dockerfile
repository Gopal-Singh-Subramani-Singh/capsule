FROM python:3.11-slim

WORKDIR /app

# System deps — minimal; curl for HEALTHCHECK only
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user
RUN groupadd -g 1000 capsule && useradd -u 1000 -g capsule -s /sbin/nologin capsule

# Python deps — installed as root before switching user
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model artifact
COPY iris_model_module_optimised.onnx /app/model/iris_model_module_optimised.onnx

# Server
COPY server.py .

RUN chown -R capsule:capsule /app

USER capsule

ENV MODEL_PATH=/app/model/iris_model_module_optimised.onnx
ENV FRAMEWORK=onnx
ENV PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "server.py"]
