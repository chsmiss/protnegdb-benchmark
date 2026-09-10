#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set_name="${1:-pilot48}"
base="${project_root}/data/interim/af3_partner_ranking_v1"
protein_list="${base}/${set_name}_proteins_v1.txt"
input_dir="${base}/monomer_inputs_v1"
output_root="${base}/monomer_data_v1"
log_root="${base}/monomer_logs_v1"
af3_python="/home/chs/miniforge3/envs/alphafold3/bin/python"
af3_runner="/data/chs/00.software/17.alphafold3/alphafold3/run_alphafold.py"
model_dir="/data/tmp/chs2scy/alphafold3_model_params"
db_dir="/data/pubres/proteinRes/MSAdb/alphafold3_database"
workers="${AF3_MSA_WORKERS:-4}"
cpu_per_search="${AF3_MSA_CPU_PER_SEARCH:-4}"

test -s "${protein_list}"
test -x "${af3_python}"
test -s "${af3_runner}"
test -s "${model_dir}/af3.bin"
test -d "${db_dir}"
mkdir -p "${output_root}" "${log_root}"

run_one() {
    accession="$1"
    input="${input_dir}/${accession}.json"
    destination="${output_root}/${accession}"
    log="${log_root}/${accession}.log"
    done_json="${destination}/AF3MONO_${accession}/AF3MONO_${accession}_data.json"
    if [[ -s "${done_json}" ]]; then
        printf 'SKIP\t%s\n' "${accession}"
        return 0
    fi
    mkdir -p "${destination}"
    printf 'START\t%s\n' "${accession}"
    CUDA_VISIBLE_DEVICES='' XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "${af3_python}" "${af3_runner}" \
        --json_path="${input}" \
        --model_dir="${model_dir}" \
        --db_dir="${db_dir}" \
        --output_dir="${destination}" \
        --norun_inference \
        --max_template_date=2021-09-30 \
        --jackhmmer_n_cpu="${cpu_per_search}" \
        --nhmmer_n_cpu="${cpu_per_search}" \
        --jackhmmer_binary_path="/home/chs/miniforge3/envs/alphafold3/bin/jackhmmer" \
        --nhmmer_binary_path="/home/chs/miniforge3/envs/alphafold3/bin/nhmmer" \
        --hmmalign_binary_path="/home/chs/miniforge3/envs/alphafold3/bin/hmmalign" \
        --hmmsearch_binary_path="/home/chs/miniforge3/envs/alphafold3/bin/hmmsearch" \
        --hmmbuild_binary_path="/home/chs/miniforge3/envs/alphafold3/bin/hmmbuild" \
        >"${log}" 2>&1
    test -s "${done_json}"
    printf 'DONE\t%s\n' "${accession}"
}

export input_dir output_root log_root af3_python af3_runner model_dir db_dir cpu_per_search
export -f run_one

tr '\n' '\0' < "${protein_list}" | xargs -0 -r -n1 -P "${workers}" bash -c 'run_one "$1"' _

expected="$(awk 'NF {n++} END {print n+0}' "${protein_list}")"
completed="$(find "${output_root}" -mindepth 3 -maxdepth 3 -name 'AF3MONO_*_data.json' -size +0c | wc -l)"
printf 'SUMMARY\texpected=%s\tall_cached=%s\tset=%s\n' "${expected}" "${completed}" "${set_name}"
