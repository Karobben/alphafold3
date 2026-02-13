#!/bin/bash

APPDIR="/home/wenkanl2/BioTools/af3Diftest/"  # Replace with your actual path if different
ALPHAFOLD3DIR="$APPDIR/alphafold3"
#HMMER3_BINDIR="/usr/bin" # Path to HMMER binaries (**installed via OS package manager or specify your path**)
HMMER3_BINDIR="${CONDA_PREFIX}/bin/" # Path to Conda binarys (**installed via conda**)
DB_DIR="/home/wenkanl2/BioTools/alphafold3/public_databases"
MODEL_DIR="/home/wenkanl2/BioTools/alphafold3/models"
WORK_DIR=$(pwd)
OUTPUT_DIR="${WORK_DIR}/output/${BASE_NAME}"
LOG_FILE="${OUTPUT_DIR}/af3_run.log"
JSON_FILE=$(ls -1 *.json 2>/dev/null | head -n 1)

run_alphafold.py \
    --jackhmmer_binary_path="${HMMER3_BINDIR}/jackhmmer" \
    --nhmmer_binary_path="${HMMER3_BINDIR}/nhmmer" \
    --hmmalign_binary_path="${HMMER3_BINDIR}/hmmalign" \
    --hmmsearch_binary_path="${HMMER3_BINDIR}/hmmsearch" \
    --hmmbuild_binary_path="${HMMER3_BINDIR}/hmmbuild" \
    --initial_structure_pdb="Test.pdb" \
    --db_dir="${DB_DIR}" \
    --model_dir="${MODEL_DIR}" \
    --json_path="${WORK_DIR}/${JSON_FILE}" \
    --output_dir="${OUTPUT_DIR}" \
    --buckets="256,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120" \
    2>&1 | tee -a "${LOG_FILE}"
