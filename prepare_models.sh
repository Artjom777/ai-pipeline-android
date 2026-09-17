#!/usr/bin/env bash
set -euo pipefail
TARGET_DIR="${1:-/workspace/app/src/main/assets}"
mkdir -p "$TARGET_DIR/text_encoder" "$TARGET_DIR/unet" "$TARGET_DIR/vae"
TEXT_URL="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/text_encoder/model.fp16.safetensors?download=true"
UNET_URL="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors?download=true"
VAE_URL="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors?download=true"
TEXT_FILE="$TARGET_DIR/text_encoder/model.fp16.safetensors"
UNET_FILE="$TARGET_DIR/unet/diffusion_pytorch_model.fp16.safetensors"
VAE_FILE="$TARGET_DIR/vae/diffusion_pytorch_model.fp16.safetensors"
download_file(){
local url="$1"
local dest="$2"
if [ ! -f "$dest" ] || [ ! -s "$dest" ]; then
curl -L --retry 3 --retry-delay 2 -f -s -S -o "${dest}.tmp" "$url"
mv -f "${dest}.tmp" "$dest"
fi
}
download_file "$TEXT_URL" "$TEXT_FILE" &
PID_TEXT=$!
download_file "$UNET_URL" "$UNET_FILE" &
PID_UNET=$!
download_file "$VAE_URL" "$VAE_FILE" &
PID_VAE=$!
wait $PID_TEXT
wait $PID_UNET
wait $PID_VAE
test -s "$TEXT_FILE"
test -s "$UNET_FILE"
test -s "$VAE_FILE"
cp -f "$TEXT_FILE" "$TARGET_DIR/text_encoder.mnn"
cp -f "$UNET_FILE" "$TARGET_DIR/unet.mnn"
cp -f "$VAE_FILE" "$TARGET_DIR/vae_decoder.mnn"
if [ -d "/data/data/com.aipipe.app/files" ] && [ -w "/data/data/com.aipipe.app/files" ]; then
mkdir -p "/data/data/com.aipipe.app/files/text_encoder" "/data/data/com.aipipe.app/files/unet" "/data/data/com.aipipe.app/files/vae"
cp -f "$TEXT_FILE" "/data/data/com.aipipe.app/files/text_encoder/model.fp16.safetensors"
cp -f "$UNET_FILE" "/data/data/com.aipipe.app/files/unet/diffusion_pytorch_model.fp16.safetensors"
cp -f "$VAE_FILE" "/data/data/com.aipipe.app/files/vae/diffusion_pytorch_model.fp16.safetensors"
cp -f "$TARGET_DIR/text_encoder.mnn" "/data/data/com.aipipe.app/files/text_encoder.mnn"
cp -f "$TARGET_DIR/unet.mnn" "/data/data/com.aipipe.app/files/unet.mnn"
cp -f "$TARGET_DIR/vae_decoder.mnn" "/data/data/com.aipipe.app/files/vae_decoder.mnn"
fi
