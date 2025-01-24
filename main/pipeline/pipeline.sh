#!/bin/bash -i
# 0. setup
## 0.1. environment
SHORT_NAME=walker # used as directory name
ENV_NAME=Walker2d-v2 # used as file name
EXP_ID=0
HOME_DIR=/home/zichang/proj/IQ-Learn/iq_learn

## 0.2. variables: default hyperparameter values
NUM_TRAJS=100
MAX_ITERS=100
COND_DIM=10
BELIEF_SIZE=256
CODING_LEN_COEFF=0.0001
MAX_CODING_LEN_COEFF=0.0
SAVE_INTERVAL=500
BC_STEPS=1000
BC_SAVE_INTERVAL=500
KLD_ALPHA=1
AGENT_ACTOR_LR=3e-05
AGENT_INIT_TEMP=1e-12
SEED=0
AGENT=sac
ENV_LEARN_STEPS=1000000
BC_ALPHA=0.5
METHOD_LOSS="v0"
LEVEL=2
ADDITIONAL_LOSS=none
CQL_COEF=1
NUM_RANDOM=5
SAVE_LAST=True
LAST_STEP=0
STAGE_1_MODEL=""
STAGE_2_MODEL=""
STAGE_1_COND=""
STAGE_2_COND=""



# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --SHORT_NAME) SHORT_NAME="$2"; shift 2;;
        --ENV_NAME) ENV_NAME="$2"; shift 2;;
        --NUM_TRAJS) NUM_TRAJS="$2"; shift 2;;
        --MAX_ITERS) MAX_ITERS="$2"; shift 2;;
        --COND_DIM) COND_DIM="$2"; shift 2;;
        --BELIEF_SIZE) BELIEF_SIZE="$2"; shift 2;;
        --CODING_LEN_COEFF) CODING_LEN_COEFF="$2"; shift 2;;
        --MAX_CODING_LEN_COEFF) MAX_CODING_LEN_COEFF="$2"; shift 2;;
        --SAVE_INTERVAL) SAVE_INTERVAL="$2"; shift 2;;
        --BC_STEPS) BC_STEPS="$2"; shift 2;;
        --BC_SAVE_INTERVAL) BC_SAVE_INTERVAL="$2"; shift 2;;
        --KLD_ALPHA) KLD_ALPHA="$2"; shift 2;;
        --AGENT_ACTOR_LR) AGENT_ACTOR_LR="$2"; shift 2;;
        --AGENT_INIT_TEMP) AGENT_INIT_TEMP="$2"; shift 2;;
        --SEED) SEED="$2"; shift 2;;
        --AGENT) AGENT="$2"; shift 2;;
        --ENV_LEARN_STEPS) ENV_LEARN_STEPS="$2"; shift 2;;
        --BC_ALPHA) BC_ALPHA="$2"; shift 2;;
        --METHOD_LOSS) METHOD_LOSS="$2"; shift 2;;
        --LEVEL) LEVEL="$2"; shift 2;;
        --ADDITIONAL_LOSS) ADDITIONAL_LOSS="$2"; shift 2;;
        --CQL_COEF) CQL_COEF="$2"; shift 2;;
        --NUM_RANDOM) NUM_RANDOM="$2"; shift 2;;
        --SAVE_LAST) SAVE_LAST="$2"; shift 2;;
        --EXP_ID) EXP_ID="$2"; shift 2;;
        --LAST_STEP) LAST_STEP="$2"; shift 2;;
        --STAGE_1_MODEL) STAGE_1_MODEL="$2"; shift 2;;
        --STAGE_2_MODEL) STAGE_2_MODEL="$2"; shift 2;;
        --STAGE_1_COND) STAGE_1_COND="$2"; shift 2;;
        --STAGE_2_COND) STAGE_2_COND="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done
## 0.3. resuming
#   - write a record when one step is finished
#   - if the script is interrupted, check the record and resume from the last step
# if LAST_STEP is not 0, then EXP_ID should be provided, STAGE_1_MODEL and STAGE_2_MODEL are optional
if [ "$LAST_STEP" != "0" ]; then
    if [ "$EXP_ID" == "0" ]; then
        echo "Error: EXP_ID is required when LAST_STEP is not 0."
        exit 1
    fi
fi

if [ "$EXP_ID" == "0" ]; then
    EXP_ID=$(date +'%y%m%d_%H%M%S')
fi

# Define meta file path
LOG_DIR="$HOME_DIR/pipeline/logs/$SHORT_NAME/${EXP_ID}"
TRACKING_FILE="${LOG_DIR}/tracking.txt"


if [ -f "$TRACKING_FILE" ] && [ "$LAST_STEP" == "0" ]; then
    echo "Resuming from the last step"
    last_step=$(tail -n 1 $TRACKING_FILE)
    if [ $last_step -eq 11 ]; then
        echo "Step 1.1 is finished"
    elif [ $last_step -eq 12 ]; then
        echo "Step 1.2 is finished"
    elif [ $last_step -eq 21 ]; then
        echo "Step 2.1 is finished"
    elif [ $last_step -eq 22 ]; then
        echo "Step 2.2 is finished"
    elif [ $last_step -eq 23 ]; then
        echo "Step 2.3 is finished"
    elif [ $last_step -eq 31 ]; then
        echo "Step 3.1 is finished"
    elif [ $last_step -eq 32 ]; then
        echo "Step 3.2 is finished"
    elif [ $last_step -eq 33 ]; then
        echo "Step 3.3 is finished"
    fi
else
    OLD_ID=""
    if [ "$LAST_STEP" != "0" ]; then
        # Initialize OLD_ID before changing EXP_ID
        OLD_ID=$EXP_ID
        echo "Borrowing last step $LAST_STEP from previous experiment $OLD_ID."
        EXP_ID=$(date +'%y%m%d_%H%M%S')

        # Copy the models
        mkdir -p "$HOME_DIR/encoder/experiments/$SHORT_NAME/$EXP_ID"
        if [ "$STAGE_1_MODEL" != "" ]; then
            cp "$HOME_DIR/encoder/experiments/$SHORT_NAME/$OLD_ID/$STAGE_1_MODEL" "$HOME_DIR/encoder/experiments/$SHORT_NAME/$EXP_ID/"
        elif [ "$STAGE_2_MODEL" != "" ]; then
            cp "$HOME_DIR/encoder/experiments/$SHORT_NAME/$OLD_ID/$STAGE_2_MODEL" "$HOME_DIR/encoder/experiments/$SHORT_NAME/$EXP_ID/"
        else
            # Ensure we are not copying the directory into itself
            if [ "$OLD_ID" != "$EXP_ID" ]; then
                cp -r "$HOME_DIR/encoder/experiments/$SHORT_NAME/$OLD_ID/"* "$HOME_DIR/encoder/experiments/$SHORT_NAME/$EXP_ID/"
            else
                echo "Error: Source and destination directories are the same."
            fi
        fi

        # Copy the conditions
        mkdir -p "$HOME_DIR/cond/$SHORT_NAME/$EXP_ID"
        if [ "$STAGE_1_COND" != "" ]; then
            cp "$HOME_DIR/cond/$SHORT_NAME/$OLD_ID/$STAGE_1_COND" "$HOME_DIR/cond/$SHORT_NAME/$EXP_ID/"
        elif [ "$STAGE_2_COND" != "" ]; then
            cp "$HOME_DIR/cond/$SHORT_NAME/$OLD_ID/$STAGE_2_COND" "$HOME_DIR/cond/$SHORT_NAME/$EXP_ID/"
        else
            # Ensure we are not copying the directory into itself
            if [ "$OLD_ID" != "$EXP_ID" ]; then
                cp -r "$HOME_DIR/cond/$SHORT_NAME/$OLD_ID/"* "$HOME_DIR/cond/$SHORT_NAME/$EXP_ID/"
            else
                echo "Error: Source and destination directories are the same."
            fi
        fi

        # copy
    fi
    LOG_DIR="$HOME_DIR/pipeline/logs/$SHORT_NAME/${EXP_ID}"
    META_FILE="$LOG_DIR/meta.txt"
    TRACKING_FILE="${LOG_DIR}/tracking.txt"
    mkdir -p "$LOG_DIR"
    echo "Starting with EXP_ID: $EXP_ID"
    last_step=$LAST_STEP
    echo -e "${last_step}" > $TRACKING_FILE
    # Save meta information
    echo "Experiment ID: ${EXP_ID}" > "$META_FILE"
    echo "SHORT_NAME: $SHORT_NAME" >> "$META_FILE"
    echo "ENV_NAME: $ENV_NAME" >> "$META_FILE"
    echo "NUM_TRAJS: $NUM_TRAJS" >> "$META_FILE"
    echo "MAX_ITERS: $MAX_ITERS" >> "$META_FILE"
    echo "COND_DIM: $COND_DIM" >> "$META_FILE"
    echo "BELIEF_SIZE: $BELIEF_SIZE" >> "$META_FILE"
    echo "CODING_LEN_COEFF: $CODING_LEN_COEFF" >> "$META_FILE"
    echo "MAX_CODING_LEN_COEFF: $MAX_CODING_LEN_COEFF" >> "$META_FILE"
    echo "SAVE_INTERVAL: $SAVE_INTERVAL" >> "$META_FILE"
    echo "BC_STEPS: $BC_STEPS" >> "$META_FILE"
    echo "BC_SAVE_INTERVAL: $BC_SAVE_INTERVAL" >> "$META_FILE"
    echo "KLD_ALPHA: $KLD_ALPHA" >> "$META_FILE"
    echo "AGENT_ACTOR_LR: $AGENT_ACTOR_LR" >> "$META_FILE"
    echo "AGENT_INIT_TEMP: $AGENT_INIT_TEMP" >> "$META_FILE"
    echo "SEED: $SEED" >> "$META_FILE"
    echo "AGENT: $AGENT" >> "$META_FILE"
    echo "ENV_LEARN_STEPS: $ENV_LEARN_STEPS" >> "$META_FILE"
    echo "BC_ALPHA: $BC_ALPHA" >> "$META_FILE"
    echo "METHOD_LOSS: $METHOD_LOSS" >> "$META_FILE"
    echo "LEVEL: $LEVEL" >> "$META_FILE"
    echo "ADDITIONAL_LOSS: $ADDITIONAL_LOSS" >> "$META_FILE"
    echo "CQL_COEF: $CQL_COEF" >> "$META_FILE"
    echo "NUM_RANDOM: $NUM_RANDOM" >> "$META_FILE"
    echo "SAVE_LAST: $SAVE_LAST" >> "$META_FILE"
    if [ "$LAST_STEP" != "0" ]; then
        echo "" >> "$META_FILE"
        echo "LAST_STEP: $LAST_STEP" >> "$META_FILE"
        echo "STAGE_1_MODEL: $STAGE_1_MODEL" >> "$META_FILE"
        echo "STAGE_2_MODEL: $STAGE_2_MODEL" >> "$META_FILE"
        echo "STAGE_1_COND: $STAGE_1_COND" >> "$META_FILE"
        echo "STAGE_2_COND: $STAGE_2_COND" >> "$META_FILE"
        echo "Borrowed Experiment ID: $OLD_ID" >> "$META_FILE"
    fi
fi

# Echo names directly
echo "SHORT_NAME: $SHORT_NAME"
echo "ENV_NAME: $ENV_NAME"

GREEN='\033[0;32m'
NC='\033[0m' # No Color
echo -e \
"$GREEN
██████╗ ██╗██████╗ ███████╗██╗     ██╗███╗   ██╗███████╗    ██╗   ██╗ ██╗
██╔══██╗██║██╔══██╗██╔════╝██║     ██║████╗  ██║██╔════╝    ██║   ██║███║
██████╔╝██║██████╔╝█████╗  ██║     ██║██╔██╗ ██║█████╗      ██║   ██║╚██║
██╔═══╝ ██║██╔═══╝ ██╔══╝  ██║     ██║██║╚██╗██║██╔══╝      ╚██╗ ██╔╝ ██║
██║     ██║██║     ███████╗███████╗██║██║ ╚████║███████╗     ╚████╔╝  ██║
╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝      ╚═══╝   ╚═╝$NC"




## 0.4. activate environment

cd $HOME_DIR
source activate base
conda activate hil

echo \
"
███████╗████████╗ █████╗  ██████╗ ███████╗     ██╗
██╔════╝╚══██╔══╝██╔══██╗██╔════╝ ██╔════╝    ███║
███████╗   ██║   ███████║██║  ███╗█████╗      ╚██║
╚════██║   ██║   ██╔══██║██║   ██║██╔══╝       ██║
███████║   ██║   ██║  ██║╚██████╔╝███████╗     ██║
╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝     ╚═╝"
# Stage 1. Deterministic encoder training

## 1.1. Deterministic encoder training
#   - load 300 trajs to encoder
if [ $last_step -lt 11 ]; then
    echo "Step 1.1: Deterministic encoder training"
    # find training expert data
    filtered_files=$(ls experts/$SHORT_NAME/ | grep "${ENV_NAME}_${NUM_TRAJS}_")
    sorted_files=$(echo "$filtered_files" | sort -t '_' -k3n)
    if [ $(echo "$sorted_files" | wc -l) -lt 3 ]; then
        echo "Error: There are fewer than 3 files for training in experts/$SHORT_NAME/."
        exit 1
    fi
    train_data=$(echo "$sorted_files" | awk 'BEGIN {sep=""} NR<=3 { printf "%s../experts/'$SHORT_NAME'/%s",sep,$0; sep="," }')
    
    # find eval expert data
    filtered_files=$(ls experts/$SHORT_NAME/ | grep "${ENV_NAME}_10_")
    sorted_files=$(echo "$filtered_files" | sort -t '_' -k3n)
    if [ $(echo "$sorted_files" | wc -l) -lt 3 ]; then
        echo "Error: There are fewer than 3 files for evaluation in experts/$SHORT_NAME/."
        exit 1
    fi
    eval_data=$(echo "$sorted_files" | awk 'BEGIN {sep=""} NR<=3 { printf "%s../experts/'$SHORT_NAME'/%s",sep,$0; sep="," }')

    EXP_DIR="$HOME_DIR/encoder/experiments/$SHORT_NAME/$EXPERIMENT_ID"
    mkdir -p "$EXP_DIR"
    echo "Training started with 3 expert files: $train_data. Save directory: $EXP_DIR"
    
    cd $HOME_DIR/encoder
    PYOPENGL_PLATFORM=egl python train_rl.py --expert_file=$train_data --eval_expert_file=$eval_data --belief-size=$BELIEF_SIZE --name=$SHORT_NAME --coding_len_coeff=$CODING_LEN_COEFF --max_coding_len_coeff=$MAX_CODING_LEN_COEFF --hil-seq-size=1000 --kl_coeff=0.0 --rec_coeff=1.0 --use_abs_pos_kl=1.0 --batch-size=16 --dataset-path=$SHORT_NAME --max-iters=$MAX_ITERS --save_interval=$SAVE_INTERVAL --state-size=8 --use_min_length_boundary_mask --latent-n=10 --exp_id=$EXP_ID &
    wait
    if [ $? -eq 0 ]; then
        echo -e "11" >> $TRACKING_FILE
    else
        echo "Step 1.1 failed"
        exit 1
    fi
fi

## 1.2. Evaluate the deterministic encoder
#   - save dir is in the model training path
if [ $last_step -lt 12 ]; then
    echo "Step 1.2: Evaluate the deterministic encoder"
    cd $HOME_DIR

    directory="encoder/experiments/$SHORT_NAME/$EXP_ID/"
    stage1_model=$(find "$directory" -type f -name "model-*.ckpt" | sort -V | tail -n 1 | xargs basename)
    if [ -z "$stage1_model" ]; then
        echo "Error: No file containing 'model-' found in directory $directory."
        exit 1
    else
        echo "Model file found: $stage1_model"
    fi

    filtered_files=$(ls experts/$SHORT_NAME/ | grep "${ENV_NAME}_10_")
    sorted_files=$(echo "$filtered_files" | sort -t '_' -k3n)
    if [ $(echo "$sorted_files" | wc -l) -lt 3 ]; then
        echo "Error: There are fewer than 3 files in experts/$SHORT_NAME/."
        exit 1
    else
        first_three_files=$(echo "$sorted_files" | awk 'BEGIN {sep=""} NR<=3 { printf "%s../experts/'$SHORT_NAME'/%s",sep,$0; sep="," }')
        echo "Evaluation of deterministic encoder started with 3 expert files: $first_three_files. Plot can be found at encoder/plot/$SHORTNAME/test_stage1_PCA.png later."
    fi

    cd $HOME_DIR/encoder
    python embed_hil.py test_stage1 $first_three_files -b env=\"$SHORT_NAME\" -b checkpoint=\"experiments/$SHORT_NAME/$EXP_ID/$stage1_model\" --exp_id=$EXP_ID &
    wait
    if [ $? -eq 0 ]; then
        echo -e "12" >> $TRACKING_FILE
    else
        echo "Step 1.2 failed"
        exit 1
    fi
fi

echo \
"
███████╗████████╗ █████╗  ██████╗ ███████╗    ██████╗ 
██╔════╝╚══██╔══╝██╔══██╗██╔════╝ ██╔════╝    ╚════██╗
███████╗   ██║   ███████║██║  ███╗█████╗       █████╔╝
╚════██║   ██║   ██╔══██║██║   ██║██╔══╝      ██╔═══╝ 
███████║   ██║   ██║  ██║╚██████╔╝███████╗    ███████╗
╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚══════╝"
# Stage 2. Probabilistic encoder training
## 2.1. condition generation
#   - generate 20 expert trajs by combining 2 skill levels
#   - embed the expert trajs to get the condition
if [ $last_step -lt 21 ]; then
    echo "Step 2.1. condition generation" 
    cd $HOME_DIR
    directory="encoder/experiments/$SHORT_NAME/$EXP_ID/"
    stage1_model=$(find "$directory" -type f -name "model-*.ckpt" | sort -V | tail -n 1 | xargs basename)
    if [ -z "$stage1_model" ]; then
        echo "Error: No file containing 'model-' found in directory $directory."
        exit 1
    else
        echo "Model file found: $stage1_model"
    fi

    filtered_files=$(ls experts/$SHORT_NAME/ | grep "${ENV_NAME}_10_")
    sorted_files=$(echo "$filtered_files" | sort -t '_' -k3n)
    file_count=$(echo "$sorted_files" | wc -l)

    if [ $file_count -lt $LEVEL ]; then
        echo "Error: There are fewer than $LEVEL files in experts/$SHORT_NAME/."
        exit 1
    else
        if [ $LEVEL -eq 2 ]; then
            # Select the first and last files from the sorted list
            first_file=$(echo "$sorted_files" | head -n 1)
            last_file=$(echo "$sorted_files" | tail -n 1)

            # Combine the file paths with proper formatting
            selected_files=$(echo -e "$first_file\n$last_file" | awk -v short_name="$SHORT_NAME" 'BEGIN {sep=""} { printf "%s../experts/" short_name "/%s", sep, $0; sep="," }')

            # Extract the last words from the first and last files
            selected_words=$(echo -e "$first_file\n$last_file" | awk -F'_' '{gsub(/\..+$/, "", $NF); print $NF}')

            # Combine the last words with '+'
            combined_words=$(echo "$selected_words" | paste -sd '+')

            echo "Combining $LEVEL skill levels: $selected_files"
        elif [ $LEVEL -eq 3 ]; then
            # Select the first, second, and last files from the sorted list
            first_file=$(echo "$sorted_files" | head -n 1)
            second_file=$(echo "$sorted_files" | sed -n '2p')
            last_file=$(echo "$sorted_files" | tail -n 1)

            # Combine the file paths with proper formatting
            selected_files=$(echo -e "$first_file\n$second_file\n$last_file" | awk -v short_name="$SHORT_NAME" 'BEGIN {sep=""} { printf "%s../experts/" short_name "/%s", sep, $0; sep="," }')

            # Extract the last words from the first, second, and last files
            selected_words=$(echo -e "$first_file\n$second_file\n$last_file" | awk -F'_' '{gsub(/\..+$/, "", $NF); print $NF}')

            # Combine the last words with '+'
            combined_words=$(echo "$selected_words" | paste -sd '+')

            echo "Combining 3 skill levels: $selected_files"
        fi
    fi

    traj_name=${ENV_NAME}_${LEVEL}0_${combined_words}

    cd $HOME_DIR/utils
    python combine_data.py $selected_files 10 ../experts/$SHORT_NAME/${traj_name}.pkl &
    wait
    echo "Combined $LEVEL skill levels to experts/$SHORT_NAME/$traj_name.pkl"

    cd $HOME_DIR/encoder
    python embed_hil.py $traj_name ../experts/$SHORT_NAME/${traj_name}.pkl -b env=\"$SHORT_NAME\" -b checkpoint=\"experiments/$SHORT_NAME/$EXP_ID/$stage1_model\" --embed_mode=det --exp_id=$EXP_ID &
    wait

    if [ $? -eq 0 ]; then
        echo -e "21" >> $TRACKING_FILE
        echo "Embedded the combined traj to condition cond/$SHORT_NAME/$EXP_ID/${traj_name}.pkl"
    else
        echo "Step 2.1 failed"
        exit 1
    fi

fi
## 2.2. Probabilistic encoder training
#   - load 20 trajs to encoder
if [ $last_step -lt 22 ]; then
    echo "Step 2.2. Probabilistic encoder training"
    cd $HOME_DIR
    directory="encoder/experiments/$SHORT_NAME/$EXP_ID/"
    stage1_model=$(find "$directory" -type f -name "model-*.ckpt" | sort -V | tail -n 1 | xargs basename)
    if [ -z "$stage1_model" ]; then
        echo "Error: No file containing 'model-' found in directory $directory."
        exit 1
    else
        echo "Model file found: $stage1_model"
    fi

    demo=$(ls experts/$SHORT_NAME/ | grep "${ENV_NAME}_${LEVEL}0_" | head -n 1)
    cond=$(ls cond/$SHORT_NAME/$EXP_ID/ | grep "${ENV_NAME}_${LEVEL}0_" | head -n 1)
    echo "Training started with 20 expert trajs: $demo and initial condition: $cond"
    python train_iq.py bc_steps=$BC_STEPS bc_save_interval=$BC_SAVE_INTERVAL cond_dim=$COND_DIM method.kld_alpha=$KLD_ALPHA agent.actor_lr=$AGENT_ACTOR_LR agent.init_temp=$AGENT_INIT_TEMP seed=$SEED wandb=True env=$SHORT_NAME agent=$AGENT expert.demos=${LEVEL}0 env.learn_steps=$ENV_LEARN_STEPS method.enable_bc_actor_update=False method.bc_init=True method.bc_alpha=$BC_ALPHA env.eval_interval=1e4 cond_type=debug env.demo=$SHORT_NAME/$demo env.cond=$SHORT_NAME/$EXP_ID/$cond method.loss=$METHOD_LOSS method.regularize=True exp_dir=$HOME_DIR/encoder/experiments/$SHORT_NAME/$EXP_ID/ encoder=$stage1_model num_levels=$LEVEL &
    wait
    if [ $? -eq 0 ]; then
        echo -e "22" >> $TRACKING_FILE
    else
        echo "Step 2.2 failed"
        exit 1
    fi
fi

## 2.3. Probabilistic encoder evaluation
if [ $last_step -lt 23 ]; then
    echo "Step 2.3. Probabilistic encoder evaluation"
    cd $HOME_DIR/encoder
    prob_encoder=$(find "experiments/$SHORT_NAME/$EXP_ID/" -type f -name "*prob-encoder*" | sort -t_ -k7n | tail -n 1)
    if [ -z "$prob_encoder" ]; then
        echo "Error: No file containing 'prob-encoder' found in directory encoder/experiments/$SHORT_NAME/$EXP_ID/."
        exit 1
    else
        echo "Probabilistic encoder found: $prob_encoder"
    fi

    # Pick the expert files based on the skill level
    cd $HOME_DIR
    filtered_files=$(ls experts/$SHORT_NAME/ | grep "${ENV_NAME}_10_")
    sorted_files=$(echo "$filtered_files" | sort -t '_' -k3n)
    file_count=$(echo "$sorted_files" | wc -l)

    if [ $file_count -lt $LEVEL ]; then
        echo "Error: There are fewer than $LEVEL files in experts/$SHORT_NAME/."
        exit 1
    else
        if [ $LEVEL -eq 2 ]; then
            # Select the first and last files (low and high skill levels)
            first_file=$(echo "$sorted_files" | head -n 1)
            last_file=$(echo "$sorted_files" | tail -n 1)
            selected_files=$(printf "../experts/%s/%s,../experts/%s/%s" "$SHORT_NAME" "$first_file" "$SHORT_NAME" "$last_file")
            echo "Evaluation of mean as embeddings started with 2 expert files: $selected_files"
        elif [ $LEVEL -eq 3 ]; then
            # Select three files (low, medium, and high skill levels)
            first_file=$(echo "$sorted_files" | head -n 1)
            mid_file=$(echo "$sorted_files" | sed -n '2p')
            last_file=$(echo "$sorted_files" | tail -n 1)
            selected_files=$(printf "../experts/%s/%s,../experts/%s/%s,../experts/%s/%s" "$SHORT_NAME" "$first_file" "$SHORT_NAME" "$mid_file" "$SHORT_NAME" "$last_file")
            echo "Evaluation of mean as embeddings started with 3 expert files: $selected_files"
        fi
    fi


    cd $HOME_DIR/encoder
    prob_encoder_purename=$(basename "$prob_encoder" .ckpt)
    python embed_hil.py test_meanAsEmb_$prob_encoder_purename $selected_files -b env=\"$SHORT_NAME\" -b checkpoint=\"$prob_encoder\" --embed_mode=mean --n_features=$LEVEL --exp_id=$EXP_ID &
    wait
    if [ $? -eq 0 ]; then
        echo -e "23" >> $TRACKING_FILE
        echo "Evaluated using mean as embedding. Plot can be found at encoder/plot/$SHORTNAME/$EXP_ID/test_meanAsEmb_${prob_encoder_purename}_PCA.png"
    else
        echo "Step 2.3 failed"
        exit 1
    fi
fi

echo \
"
███████╗████████╗ █████╗  ██████╗ ███████╗    ██████╗ 
██╔════╝╚══██╔══╝██╔══██╗██╔════╝ ██╔════╝    ╚════██╗
███████╗   ██║   ███████║██║  ███╗█████╗       █████╔╝
╚════██║   ██║   ██╔══██║██║   ██║██╔══╝       ╚═══██╗
███████║   ██║   ██║  ██║╚██████╔╝███████╗    ██████╔╝
╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═════╝ "
# Stage 3. Decoder training
## 3.1. Condition generation
if [ $last_step -lt 31 ]; then
    echo "Step 3.1. Condition generation"
    cd $HOME_DIR/encoder
    demo=$(find "../experts/$SHORT_NAME/" -type f -name "${ENV_NAME}_${LEVEL}0_*" | head -n 1)
    demo_purename=$(basename "$demo" .pkl)
    prob_encoder=$(find "experiments/$SHORT_NAME/$EXP_ID/" -type f -name "*prob-encoder*" | sort -t_ -k7n | tail -n 1)
    if [ -z "$demo" ]; then
        echo "Error: No file containing '${LEVEL}0' found in directory experts/$SHORT_NAME/."
        exit 1
    else
        echo "Demo found: $demo"
    fi
    if [ -z "$prob_encoder" ]; then
        echo "Error: No file containing 'prob-encoder' found in directory encoder/experiments/$SHORT_NAME/$EXP_ID/."
        exit 1
    else
        prob_encoder_purename=$(basename "$prob_encoder" .ckpt)
        echo "Probabilistic encoder found: $prob_encoder"
    fi
    cond_name=${demo_purename}_meanAsEmb_${prob_encoder_purename}
    cd $HOME_DIR/encoder
    python embed_hil.py $cond_name $demo -b env=\"$SHORT_NAME\" -b checkpoint=\"$prob_encoder\" --embed_mode=mean --exp_id=$EXP_ID &
    wait
    if [ $? -eq 0 ]; then
        echo -e "31" >> $TRACKING_FILE
        echo "Condition generated using mean as embedding. Plot can be found at encoder/plot/$SHORTNAME/$EXP_ID/${cond_name}_PCA.png"
    else
        echo "Step 3.1 failed"
        exit 1
    fi
fi

## 3.2. Decoder training
if [ $last_step -lt 32 ]; then
    echo "Step 3.2. Decoder training"
    cd $HOME_DIR
    demo=$(ls experts/$SHORT_NAME/ | grep "${ENV_NAME}_${LEVEL}0_" | head -n 1)
    demo_name=$(basename "$demo" .pkl)
    prob_encoder=$(find "encoder/experiments/$SHORT_NAME/$EXP_ID/" -type f -name "*prob-encoder*" | sort -t_ -k7n | tail -n 1)
    prob_encoder=$(basename "$prob_encoder")
    prob_encoder_name=$(basename "$prob_encoder" .ckpt)
    cond=$(ls cond/$SHORT_NAME/$EXP_ID/ | grep "${demo_name}_meanAsEmb_${prob_encoder_name}" | head -n 1)
    if [ -z "$cond" ]; then
        echo "Error: No condition containing '${demo_name}_meanAsEmb_${prob_encoder_name}' found in directory cond/$SHORT_NAME/$EXP_ID/."
        exit 1
    else
        echo -e "Cond found: $cond.\nDecoder training will start soon."
    fi

    python train_iq.py additional_loss=$ADDITIONAL_LOSS cql_coef=$CQL_COEF num_random=$NUM_RANDOM env.learn_steps=$ENV_LEARN_STEPS cond_dim=$COND_DIM method.kld_alpha=$KLD_ALPHA agent.actor_lr=$AGENT_ACTOR_LR agent.init_temp=$AGENT_INIT_TEMP seed=$SEED wandb=True env=$SHORT_NAME agent=$AGENT expert.demos=${LEVEL}0 method.enable_bc_actor_update=False method.bc_init=False method.bc_alpha=$BC_ALPHA env.eval_interval=1e4 cond_type=debug env.demo=$SHORT_NAME/$demo env.cond=$SHORT_NAME/$EXP_ID/$cond method.loss=$METHOD_LOSS method.regularize=True exp_dir=$HOME_DIR/encoder/experiments/$SHORT_NAME/$EXP_ID/ encoder=$prob_encoder num_levels=$LEVEL save_last=$SAVE_LAST &
    python_pid=$!

    # Echo the PID of the Python process
    echo "Decoder training started with PID: $python_pid"

    # Wait for the Python process to finish
    wait

    # Check the exit status of the Python process
    if [ $? -eq 0 ]; then
        echo -e "32" >> $TRACKING_FILE
    else
        echo "Step 3.2 failed"
        exit 1
    fi
fi

## 3.3 Decoder evaluation
if [ $last_step -lt 33 ]; then
    echo "TODO: Evaluate the decoder" # TODO: Evaluate the decoder
    exit 0
    if [ $? -eq 0 ]; then
        echo -e "33" >> $TRACKING_FILE
    else
        echo "Step 3.3 failed"
        exit 1
    fi
fi