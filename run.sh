mkdir -p out
docker run -d \
  --name my-gpu-app \
  --gpus all \
  --restart on-failure \
  -e HF_TOKEN="$HF_TOKEN" \
  -e PYTHONPATH=/app/src \
  -v "$(pwd)/src:/app/src:ro" \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/years.txt:/app/years.txt:ro" \
  -v "$(pwd)/out:/app/out" \
  jesse-embeddings:latest