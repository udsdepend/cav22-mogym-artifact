#!/bin/bash

mkdir build

rm -f cav22-mogym.image.tar

# Export all dependencies of the `mgym` package.
poetry export -E all --without-hashes -f requirements.txt > build/requirements.txt

echo ">>> ⏳ Building docker image..."
# Build an `amd64` image regardless of the host platform.
docker buildx build -t cav22-mogym --platform linux/amd64 .

# Save the image as a `.tar` file.
echo ">>> ⏳ Saving docker image..."
docker save --output cav22-mogym.image.tar cav22-mogym
