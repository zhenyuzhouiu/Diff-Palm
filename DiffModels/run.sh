set -ex

CHK_DIR=./checkpoint/diffusion-netpalm-scale-128

if [ ! -d "$CHK_DIR" ]; then
    mkdir -p "$CHK_DIR"
fi

export OPENAI_LOGDIR=$CHK_DIR

MODEL_FLAGS="--large_size 128 --small_size 128 --in_channels 4 --out_channels 3 --num_channels 64 --num_res_blocks 2 --learn_sigma True --dropout 0.2 --attention_resolutions 4 --class_cond False"

DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"

TRAIN_FLAGS="--lr 2e-4 --batch_size 64  --save_interval 10000"

DATA_DIR=""

RESUME=""

python ./palm_train.py \
    --data_dir $DATA_DIR $TRAIN_FLAGS $DIFFUSION_FLAGS $MODEL_FLAGS $RESUME
