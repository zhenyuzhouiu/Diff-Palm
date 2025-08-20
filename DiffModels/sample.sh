set -ex

GPUS="0"
BATCH=20 #125
K_STEP=500

CHK_DIR=./output/test-large
export OPENAI_LOGDIR=$CHK_DIR

if [ ! -d "$CHK_DIR" ]; then
    mkdir -p "$CHK_DIR"
fi

INDIR=/home/ra1/Project/ZZY/Diff-Palm/PolyCreases/test-images/image
NPZ="${CHK_DIR}/data.npz"
OUTDIR1="${CHK_DIR}/label"
NUM=40 #2000000
SAMPLES=40 #2000000
SAME_NUM=20 #125

python3 scripts/save_npz.py \
    --input $INDIR \
    --outdir $OUTDIR1 \
    --outnpz $NPZ \
    --num ${NUM} \
    --same_num ${SAME_NUM} \


INTRA_FLAGS="--sharing_num ${SAME_NUM} --sharing_step ${K_STEP}"

SAMPLE_FLAGS="--batch_size ${BATCH} --num_samples ${SAMPLES} --use_ddim False"

MODEL_FLAGS="--large_size 128 --small_size 128 --in_channels 4 --out_channels 3 --num_channels 64 --num_res_blocks 2 --learn_sigma True --dropout 0.1 --attention_resolutions 4 --class_cond False"

DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"

MODEL_PATH="checkpoint/diffusion-netpalm-scale-128/ema_0.9999.pt"

DATA_PATH=$NPZ

# CUDA_VISIBLE_DEVICES=${GPUS}  python3 ./palm_sample_intra.py \
#     --model_path $MODEL_PATH \
#     --base_samples $DATA_PATH \
#     $SAMPLE_FLAGS $DIFFUSION_FLAGS $MODEL_FLAGS $INTRA_FLAGS

mpirun -np 2 python3 ./palm_sample_intra.py \
    --model_path $MODEL_PATH \
    --base_samples $DATA_PATH \
    $SAMPLE_FLAGS $DIFFUSION_FLAGS $MODEL_FLAGS $INTRA_FLAGS

python3 scripts/load_npz.py \
    --input "${CHK_DIR}/samples_${SAMPLES}x128x128x3.npz" \
    --outdir "${CHK_DIR}/results"
