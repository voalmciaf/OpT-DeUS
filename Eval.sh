#!/bin/bash
#SBATCH --mail-user=
#SBATCH --mail-type=
#SBATCH --partition=
#SBATCH --qos=gpu
#SBATCH --gres=gpu:
#SBATCH --mem=
#SBATCH --time=
#SBATCH --job-name=
#SBATCH --output=


#load relative modules



MODEL_BASE_DIR="/Path/To/Model"
OUTPUT_BASE_DIR="/Path/To/Output"
models=( "SOLAR" "Avg-DeUS" "OpT-DeUS" "SOLAR" "Llama-Pro" "LESA" "Base")
checkpoints=(
  "checkpoint-20%"
  "checkpoint-40%"
  "checkpoint-60%"
  "checkpoint-80%"
  "checkpoint-100%"
)


TOKENIZER_FILES_DIR="/Path/To/Tokenizer"

for model in "${models[@]}"; do
  for ckpt in "${checkpoints[@]}"; do
    echo "===== Evaluating Model: ${model}, Checkpoint: ${ckpt} ====="

    CKPT_DIR="${MODEL_BASE_DIR}/${model}/${ckpt}"
    if [ ! -d "${CKPT_DIR}" ]; then
      echo "Warning: Checkpoint directory not found: ${CKPT_DIR}"
      continue
    fi

    cp "${TOKENIZER_FILES_DIR}/special_tokens_map.json"  "${CKPT_DIR}/"
    cp "${TOKENIZER_FILES_DIR}/tokenizer.json"           "${CKPT_DIR}/"
    cp "${TOKENIZER_FILES_DIR}/tokenizer_config.json"    "${CKPT_DIR}/"

    OUTPUT_PATH="${OUTPUT_BASE_DIR}/${model}/${ckpt}"
    mkdir -p "${OUTPUT_PATH}"

    lm_eval \
      --model hf \
      --model_args pretrained="${CKPT_DIR}",attn_implementation=flash_attention_2,,dtype=bfloat16 \
      --tasks logiqa,piqa,winogrande,arc_easy,wikitext,mmlu,commonsense_qa,boolq \
      --device cuda:0 \
      --batch_size 32 \
      --num_fewshot 0 \
      --output_path "${OUTPUT_PATH}"
    echo "===== Finished Evaluating: ${model}, ${ckpt} ====="
    echo
  done
done
