#!/usr/bin/env bash

set -euo pipefail

# Template batch runner for LeRobot v2.1 datasets.
# Set these variables before running, or edit the placeholders below.
DEMO_PATH="${DEMO_PATH:-/path/to/lerobot-v2.1-dataset}"
DEMO_TYPE="${DEMO_TYPE:-lerobot}"
MODEL="${MODEL:-gemini-2.5-pro}"
BACKEND="${BACKEND:-vertex}"
PROMPTS_DIR="${PROMPTS_DIR:-vlm_annotator/prompts}"
RESULTS_DIR="${RESULTS_DIR:-results/vlm_annotations}"

if [[ "${DEMO_PATH}" == "/path/to/lerobot-v2.1-dataset" ]]; then
    echo "Set DEMO_PATH to a local LeRobot v2.1 dataset path or Hugging Face repo_id." >&2
    exit 1
fi

# Optional per-task episode ranges. Update these for your dataset.
declare -A EPISODE_RANGES
EPISODE_RANGES["BimanualPlaceAppleFromBowlIntoBin"]="${BIMANUAL_PLACE_APPLE_RANGE:-[0,39]}"
EPISODE_RANGES["BimanualPlaceFruitFromBowlIntoBin"]="${BIMANUAL_PLACE_FRUIT_RANGE:-[40,79]}"
EPISODE_RANGES["BimanualPutSpatulaOnPlateFromDryingRack"]="${BIMANUAL_PUT_SPATULA_RANGE:-[80,119]}"
EPISODE_RANGES["BimanualStoreCerealBoxUnderShelf"]="${BIMANUAL_STORE_CEREAL_RANGE:-[120,159]}"
EPISODE_RANGES["PlaceCupByCoaster"]="${PLACE_CUP_RANGE:-[160,199]}"
EPISODE_RANGES["PushTheCoasterToTheMug"]="${PUSH_COASTER_TO_MUG_RANGE:-[200,239]}"
EPISODE_RANGES["PutBananaOnSaucer"]="${PUT_BANANA_RANGE:-[240,249]}"
EPISODE_RANGES["PutKiwiInCenterOfTable"]="${PUT_KIWI_RANGE:-[250,259]}"
EPISODE_RANGES["PutMugOnSaucer"]="${PUT_MUG_RANGE:-[260,299]}"
EPISODE_RANGES["TurnCupUpsideDown"]="${TURN_CUP_RANGE:-[300,399]}"
EPISODE_RANGES["TurnMugRightsideUp"]="${TURN_MUG_RANGE:-[400,499]}"

mkdir -p "${RESULTS_DIR}"

for cfg in "${PROMPTS_DIR}"/*.json; do
    base_cfg="$(basename "${cfg}")"
    task_name="${base_cfg%.json}"
    out_dir="${RESULTS_DIR}/${task_name}"

    if [[ -d "${out_dir}" ]]; then
        echo "[INFO] Output dir exists for ${task_name}, skipping."
        continue
    fi

    mkdir -p "${out_dir}"
    episode_range="${EPISODE_RANGES[${task_name}]:-[0,0]}"

    echo "[INFO] Running annotator for ${task_name}"
    python -m vlm_annotator.gemini_annotator \
        --demo_path "${DEMO_PATH}" \
        --demo_type "${DEMO_TYPE}" \
        --prompt_config "${cfg}" \
        --model "${MODEL}" \
        --backend "${BACKEND}" \
        --episode_idx "${episode_range}" \
        --few_shot \
        --output_dir "${out_dir}"
done

echo "[INFO] All tasks finished."
