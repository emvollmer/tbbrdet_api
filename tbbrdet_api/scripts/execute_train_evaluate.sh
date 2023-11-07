#!/bin/bash


# Function to handle optional flags
function handle_optional_flag {
    local flag_name="$1"
    local -a values=("${!2}")  # Accept an array of values

    if [[ ${#values[@]} -gt 0 ]]; then
        cmd="$flag_name ${values[@]}"
    else
	    cmd=""
    fi
    echo $cmd
}


WORKING_DIR=$(pwd)
# Change directory to submodule train script
BASE_DIR="/$(pwd | awk -F "/" '{print $2}')/"
submodule_dir=$(find $BASE_DIR -type d -name ".git" -prune -o -type d -name "TBBRDet" -print -quit 2>/dev/null)
cd $submodule_dir/scripts/mmdet/

# Initialize empty cfg options array
config_path=""
model_dir=""   # this is the directory to which the model will be saved
auto_resume=""
cfg_options=()
eval=""   # this can be either "bbox" or "segm"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-path)
            config_path="$2"
            shift 2
            ;;
        --work-dir)
            model_dir="$2"
            shift 2
            ;;
        --seed)
            seed_num="$2"
            shift 2
            ;;
        --auto-resume)
            auto_resume="--auto-resume"
            shift
            ;;
        --cfg-options)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-.* ]]; do
                cfg_options="$cfg_options $1"
                shift
            done
            ;;
        --eval)
            eval="$2"
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

# ------------------- TRAINING
# CMD: python train.py <CONFIG_PATH> --work-dir <MODEL_DIR>
# --seed <SEED_NUM> --deterministic --cfg-options 'data_root'='/path/to/datasets' 'load_from'='/path/to/pretrained/weights.pth'
training_cmd="python train.py $config_path --work-dir $model_dir --seed $seed_num --deterministic $auto_resume"

cfg_options_cmd=$(handle_optional_flag "--cfg-options" cfg_options[@])
training_cmd="$training_cmd $cfg_options_cmd"

# Run nvidia-smi in the background and redirect its output to gpu_monitoring.log
timestamp=$(date +"%Y%m%d_%H%M%S")
gpu_monitoring_filename="${timestamp}_gpu_monitoring.log"
nvidia-smi --query-gpu=timestamp,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > $gpu_monitoring_filename &
# Store the process ID (PID) of the background nvidia-smi process
nvidia_smi_pid=$!
stop=0

# Execute the training command
echo "------------------------------------"
echo "Train model by executing command:"
echo "$training_cmd"
echo "------------------------------------"
eval "$training_cmd"

# Check the exit status, if training was Killed or interrupted by keyboard for example
if [ $? -ne 0 ]; then
    echo "Training was terminated with a non-zero exit status."
    stop=1
fi

# Stop the nvidia-smi process
kill "$nvidia_smi_pid"

# Move gpu monitoring results into the most recent timestamp directory
if [ -e $gpu_monitoring_filename ]; then
    mv $gpu_monitoring_filename "$model_dir/"
    echo "Moved gpu monitoring log '$gpu_monitoring_filename' to $model_dir"
fi

# Stop the script run if the process had a non-zero exit status
if [ $stop -eq 1 ] ; then
    if [ "$0" != "$BASH_SOURCE" ]; then
        return 1
    else
        exit 1
    fi
fi

# ------------------- EVALUATION
# CMD: python test.py <CONFIG_PATH> <MODEL_PATH> --work-dir /path/to/results/ --out
# /path/to/model_eval.pickle --eval <METRIC>
eval_dir="$model_dir/evaluation"

if [ ! -d "$eval_dir" ]; then
    mkdir -p "$eval_dir"
fi

# Define evaluation command
# find the most recent "best" model path (there may be multiple if training was resumed)
model_path=$(find $model_dir -type f -name best* -not -path "*/.*" -exec ls -lt {} + | head -n 1 | awk '{print $NF}')
echo "Found model_path at $model_path"
if [[ ! -f $model_path ]]; then
    echo "No best model .pth in model directory $model_dir! No evaluation possible..."
    if [ "$0" != "$BASH_SOURCE" ]; then
        return 1
    else
        exit 1
    fi
fi

# find the most recent config file in the model directory
model_config_path=$(find "$model_dir" -type f -name "*.py" -not -path "*/.*" -exec ls -lt {} + | head -n 1 | awk '{print $NF}')

epoch_number=$(echo "$model_path" | sed -n 's/.*_epoch_\([0-9]*\)\.pth/\1/p')
eval_file='${eval_dir}/best_AR@1000_epoch_${epoch_number}-eval_${eval}.pickle'
echo "Creating evaluation file at $eval_file"

# note: Potentially change config_path to the path of the config in the model_dir
eval_cmd="python test.py $model_config_path $model_path --work-dir $eval_dir --out $eval_file --eval $eval"

# Execute the evaluation script as a module
echo "------------------------------------"
echo "Evaluate model by executing command:"
echo "$eval_cmd"
echo "------------------------------------"
eval "$eval_cmd"

# Check the exit status, if evaluation was Killed or interrupted by keyboard for example
if [ $? -ne 0 ]; then
    echo "Evaluation was terminated with a non-zero exit status."
    if [ "$0" != "$BASH_SOURCE" ]; then
        return 1
    else
        exit 1
    fi
fi

cd $WORKING_DIR