#!/usr/bin/env bash
# Download frozen layer-representation arrays from the GitHub Release.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tag="${PROTNEGDB_BENCHMARK_RELEASE_TAG:-v1.0.0}"

if [[ -n "${PROTNEGDB_BENCHMARK_REPO:-}" ]]; then
    repo="${PROTNEGDB_BENCHMARK_REPO}"
elif git -C "${root}" remote get-url origin >/dev/null 2>&1; then
    repo="$(git -C "${root}" remote get-url origin)"
    repo="${repo%.git}"
    repo="${repo#git@github.com:}"
    repo="${repo#https://github.com/}"
    repo="https://github.com/${repo#github.com/}"
    repo="${repo%/}"
else
    repo="https://github.com/chsmiss/protnegdb-benchmark"
fi

base="${repo}/releases/download/${tag}"

declare -A files=(
    ["mint_esm2_layer_cls_v1.npz"]="data/interim/mint_layerwise_probes_v1/esm2/layer_cls_v1.npz"
    ["mint_mint_layer_cls_v1.npz"]="data/interim/mint_layerwise_probes_v1/mint/layer_cls_v1.npz"
    ["plminteract_esm2_layer_cls_v1.npz"]="data/interim/plminteract_layerwise_probes_v1/esm2/layer_cls_v1.npz"
    ["plminteract_plminteract_layer_cls_v1.npz"]="data/interim/plminteract_layerwise_probes_v1/plminteract/layer_cls_v1.npz"
)

checksums="${root}/SHA256SUMS"
if [[ ! -s "${checksums}" ]]; then
    echo "Missing ${checksums}" >&2
    exit 1
fi

download() {
    local url="$1"
    local dest="$2"
    mkdir -p "$(dirname "${dest}")"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 5 --retry-delay 2 "${url}" -o "${dest}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${dest}" "${url}"
    else
        echo "curl or wget is required" >&2
        exit 1
    fi
}

expected_hash() {
    local rel="$1"
    awk -v rel="${rel}" '$2 == rel {print $1; found=1; exit} END {if (!found) exit 1}' "${checksums}"
}

for asset in "${!files[@]}"; do
    rel="${files[${asset}]}"
    dest="${root}/${rel}"
    expected="$(expected_hash "${rel}")"
    if [[ -s "${dest}" ]]; then
        actual="$(sha256sum "${dest}" | awk '{print $1}')"
        if [[ "${actual}" == "${expected}" ]]; then
            echo "OK ${rel}"
            continue
        fi
        echo "Checksum mismatch for existing ${rel}; re-downloading"
    fi
    echo "Downloading ${asset}"
    download "${base}/${asset}" "${dest}"
    actual="$(sha256sum "${dest}" | awk '{print $1}')"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "Checksum failed for ${rel}" >&2
        echo "expected ${expected}" >&2
        echo "actual   ${actual}" >&2
        exit 1
    fi
    echo "OK ${rel}"
done
