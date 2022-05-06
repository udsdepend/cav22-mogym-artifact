#!/bin/bash

rm -rf build/artifact/
rm -rf build/artifact.zip

./build-docker-image.sh

mkdir -p build/artifact/

cp -rp cav22-mogym.image.tar build/artifact/
cp -rp vendor build/artifact/
cp README.md build/artifact/
cp LICENSE build/artifact/
cp paper.pdf build/artifact/

cd build/artifact
zip -r ../artifact.zip .
