#!/bin/bash
# Compile GLSL shaders to SPIR-V
# Requires glslangValidator from Vulkan SDK
SHADER_DIR="crates/ferrite-render/shaders"
OUT_DIR="crates/ferrite-render/shaders/compiled"
mkdir -p "$OUT_DIR"

for shader in "$SHADER_DIR"/*.comp.glsl; do
    name=$(basename "$shader" .glsl)
    glslangValidator -V "$shader" -o "$OUT_DIR/$name.spv" \
        --target-env vulkan1.3 \
        -DWORKGROUP_SIZE_X=8 -DWORKGROUP_SIZE_Y=8
    echo "Compiled: $name.spv"
done
