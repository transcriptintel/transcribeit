#!/usr/bin/env bash
set -euo pipefail

install_root="${1:-${SHERPA_ONNX_INSTALL_ROOT:-vendor}}"
case "$install_root" in
  /*) ;;
  *) install_root="$(pwd -P)/$install_root" ;;
esac

case "$(uname -s)-$(uname -m)" in
  Darwin-x86_64) archive_suffix="osx-x64-shared-lib" ;;
  Darwin-arm64) archive_suffix="osx-arm64-shared-lib" ;;
  Linux-x86_64) archive_suffix="linux-x64-shared-lib" ;;
  Linux-aarch64) archive_suffix="linux-aarch64-shared-cpu-lib" ;;
  *)
    echo "Unsupported Sherpa platform: $(uname -s)-$(uname -m)" >&2
    exit 1
    ;;
esac

cargo run --quiet -- setup --component sherpa-libs --output-dir "$install_root"
lib_dir="$install_root/sherpa-onnx-v1.13.4-$archive_suffix/lib"
test -d "$lib_dir" || {
  echo "Sherpa library directory was not created: $lib_dir" >&2
  exit 1
}

echo "SHERPA_ONNX_LIB_DIR=$lib_dir"
if [[ -n "${GITHUB_ENV:-}" ]]; then
  printf 'SHERPA_ONNX_LIB_DIR=%s\n' "$lib_dir" >> "$GITHUB_ENV"
fi
