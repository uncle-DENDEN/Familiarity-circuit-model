#!/bin/bash

# define seeds and directories
weight_paths=(".")
response_paths=("SMT-BCM_response_correct/r_pre_rx1_1000ms_mean_5_4x10_50_64_8_8_mix50%_all.npy")
out_paths=("SMT-BCM_Jac_v3/pre")

# loop through each seed and directory pair
for i in {0..0}; do
  wp=${weight_paths[i]}
  rp=${response_paths[i]}
  op=${out_paths[i]}

  logfile=/user_data/weifanw/familiarity/runfind-$i.txt

  echo "Running find eigenmode $i: weight ${wp}; response ${rp}"

  srun -p gpu --gpus=1 --mem-per-gpu=12GB --mem=80GB --time=2-00:00:00 --pty -n1 bash << EOF &> $logfile &
    module load anaconda3
    source activate fam
    cd /user_data/weifanw/familiarity
    python find_eigenmode.py --weight_path ${wp} --response_path ${rp} --out_path ${op}
EOF
done

# wait for all jobs to finish
wait