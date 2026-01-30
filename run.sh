mkdir -p out
docker run -d \
  --name jesse-embeddings \
  --gpus all \
  --restart on-failure \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e PYTHONPATH=/app/src \
  -v "$(pwd)/src:/app/src:ro" \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/years.txt:/app/years.txt:ro" \
  -v "$(pwd)/out:/app/out" \
  jesse-embeddings:latest