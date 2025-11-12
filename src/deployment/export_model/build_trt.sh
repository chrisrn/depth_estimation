#!/usr/bin/env bash
set -euo pipefail

ONNX=${1:-deployment/export_model/model.onnx}
ENGINE_OUT=${2:-deployment/triton_model_repo/depth_unet_trt/1/model.plan}

trtexec --onnx=${ONNX} \
  --saveEngine=${ENGINE_OUT} \
  --fp16 \
  --minShapes=INPUT__0:1x3x224x224 \
  --optShapes=INPUT__0:4x3x224x224 \
  --maxShapes=INPUT__0:8x3x224x224 \
  --workspace=4096

echo "Saved TensorRT engine -> ${ENGINE_OUT}"
