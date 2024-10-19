#!/bin/bash
apt-get update && apt-get install -y llvm build-essential clang wget clang-format git

wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_6/nsight-systems-2024.6.1_2024.6.1.90-1_amd64.deb
apt install -y ./nsight-systems-2024.6.1_2024.6.1.90-1_amd64.deb

# Clean up downloaded .deb file
rm nsight-systems-2024.6.1_2024.6.1.90-1_amd64.deb