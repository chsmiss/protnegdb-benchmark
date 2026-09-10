#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu="${1:?GPU index 0-7 required}"
set_name="${2:-pilot48}"
if [[ ! "${gpu}" =~ ^[0-7]$ ]]; then
    echo "GPU must be an integer 0-7" >&2
    exit 2
fi
base="${project_root}/data/interim/af3_partner_ranking_v1"
input_dir="${base}/dimer_inputs_${set_name}_gpu${gpu}_v1"
output_dir="${base}/dimer_outputs_${set_name}_gpu${gpu}_seed1_triton_v1"
log="${base}/dimer_gpu${gpu}_${set_name}_seed1.log"
python_bin="/home/chs/miniforge3/envs/alphafold3_5090/bin/python"
runner="/data/chs/00.software/17.alphafold3/alphafold3/run_alphafold.py"
model_dir="/data/tmp/chs2scy/alphafold3_model_params"

test -d "${input_dir}"
test -n "$(find "${input_dir}" -maxdepth 1 -name '*.json' -print -quit)"
mkdir -p "${output_dir}"
CUDA_VISIBLE_DEVICES="${gpu}" \
XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_CLIENT_MEM_FRACTION=0.95 \
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
    "${python_bin}" "${runner}" \
    --input_dir="${input_dir}" \
    --model_dir="${model_dir}" \
    --output_dir="${output_dir}" \
    --norun_data_pipeline \
    --max_template_date=2021-09-30 \
    --num_diffusion_samples=5 \
    --flash_attention_implementation=triton \
    --buckets=256,512,768,1024 \
    >"${log}" 2>&1

expected="$(find "${input_dir}" -maxdepth 1 -name '*.json' | wc -l)"
completed="$(find "${output_dir}" -mindepth 2 -maxdepth 2 -name '*_summary_confidences.json' -size +0c | wc -l)"
echo "GPU ${gpu}: expected=${expected} completed=${completed}"
test "${completed}" -ge "${expected}"
