cd ./PolyCreases

python syn_polypalm_mp.py  --ids 20 --output test-images --nproc 16

cd ../DiffModels
bash ./sample.sh