set -ex

GPUS="0"
BATCH=200 #200
K_STEP=500

CHK_DIR=./output_zhenyu/test-large
export OPENAI_LOGDIR=$CHK_DIR

if [ ! -d "$CHK_DIR" ]; then
    mkdir -p "$CHK_DIR"
fi

INDIR=/home/ra1/Project/ZZY/Diff-Palm/PolyCreases/test-images/image
OUTDIR1="${CHK_DIR}/label"
NUM=2000000 #2000000
SAMPLES=2000000 #2000000
SAME_NUM=20 #20

python3 scripts/save_npz_zhenyu.py \
    --input $INDIR \
    --outdir $OUTDIR1 \
    --num ${NUM} \
    --same_num ${SAME_NUM} \


INTRA_FLAGS="--sharing_num ${SAME_NUM} --sharing_step ${K_STEP}"

SAMPLE_FLAGS="--batch_size ${BATCH} --num_samples ${SAMPLES} --use_ddim False"

MODEL_FLAGS="--large_size 128 --small_size 128 --in_channels 4 --out_channels 3 --num_channels 64 --num_res_blocks 2 --learn_sigma True --dropout 0.1 --attention_resolutions 4 --class_cond False"

DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"

MODEL_PATH="checkpoint/diffusion-netpalm-scale-128/ema_0.9999.pt"

DATA_PATH=$OUTDIR1
OUTDIR2="${CHK_DIR}/results"
mpirun -np 2 python3 ./palm_sample_intra_zhenyu.py \
    --model_path $MODEL_PATH \
    --base_samples $DATA_PATH \
    --outdir $OUTDIR2 \
    $SAMPLE_FLAGS $DIFFUSION_FLAGS $MODEL_FLAGS $INTRA_FLAGS

