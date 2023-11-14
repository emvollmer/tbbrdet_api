#!/bin/bash

WORKING_DIR=$(pwd)
# Change directory to submodule train script
BASE_DIR="/$(pwd | awk -F "/" '{print $2}')/"
submodule_dir=$(find $BASE_DIR -type d -name ".git" -prune -o -type d -name "TBBRDet" -print -quit 2>/dev/null)
cd "$submodule_dir"/scripts/mmdet/ || exit 1

# Initialize empty cfg options array
INPUT_FILE=""
CONFIG_PATH=""   # this is the model's config file (like "mask... .py")
CKP_PATH=""   # this is the model's checkpoint file (like "best... .pth")
CHANNEL=""
SCORE_THR=""
OUT_DIR="" # this is the optional dir to which the results will be saved (like "predictions")

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT_FILE="$2"
      shift 2
      ;;
    --config-file)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --ckp-file)
      CKP_PATH="$2"
      shift 2
      ;;
    --channel)
      CHANNEL="$2"
      shift 2
      ;;
    --score-threshold)
      SCORE_THR="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      if [ "$0" != "$BASH_SOURCE" ]; then
        return 1
      else
        exit 1
      fi
      ;;
  esac
done

# ------------------- INFERENCE
inference_cmd="python infer.py $INPUT_FILE --config-file $CONFIG_PATH --checkpoint-file $CKP_PATH -channel $CHANNEL --score-thr $SCORE_THR"

if [ -n "$OUT_DIR" ] ; then
  inference_cmd="$inference_cmd --out-dir $OUT_DIR"
fi
# Execute the inference command
stop=0
echo "------------------------------------"
echo "Infer on image with model by executing the command:"
echo "$inference_cmd"
echo "------------------------------------"
eval "$inference_cmd"

# Check the exit status, if inference was Killed or interrupted by keyboard for example
if [ $? -ne 0 ]; then
  echo "Inference was terminated with a non-zero exit status."
  stop=1
fi

# Stop the script run if the process had a non-zero exit status
if [ $stop -eq 1 ] ; then
  if [ "$0" != "$BASH_SOURCE" ]; then
    return 1
  else
    exit 1
  fi
fi

# shellcheck disable=SC2164
cd "$WORKING_DIR"
