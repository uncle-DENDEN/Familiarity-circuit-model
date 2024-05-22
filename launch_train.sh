#!/bin/bash

# define seeds and directories
tss=(1)
n_repeats=(1)

taus=(300)
gammas=(30)
sis=(100)

ses=(0)
es=(5)
wrp_paths=(".")
r_in_paths=("SMT-BCM_input_correct/r_in_cifar_all_noise_abs_5_4_10_64_8_8.npy")
theta_paths=("SMT-BCM_input_correct/BCM_theta_cifar_all_noise_abs_5_4_10_64_8_8.npy")
out_paths=("SMT-BCM_weights_tempsup")
postfixs=("mix50%_rep1_2e9pos1e7_v2")
swrps=(0)

# loop through each seed and directory pair
for i in {0..0}; do
  ts=${tss[i]}
  n_repeat=${n_repeats[0]}

  tau=${taus[i]}
  gamma=${gammas[0]}
  si=${sis[i]}

  se=${ses[0]}
  e=${es[0]}
  wrp_path=${wrp_paths[0]}
  r_in_path=${r_in_paths[0]}
  theta_path=${theta_paths[0]}
  out_path=${out_paths[i]}
  postfix=${postfixs[i]}
  swrp=${swrps[0]}

  logfile=/user_data/weifanw/familiarity/runtrain-$i.txt

  echo "Running train $i with gamma=${gamma}, interval=${tau}, ${postfix}"

  srun -p gpu --gpus=1 --mem-per-gpu=12GB --mem=40GB --time=1-00:00:00 --pty -n1 bash << EOF &> $logfile &
    module load anaconda3
    source activate fam
    cd /user_data/weifanw/familiarity
    python train.py --temporal_supervision ${ts} --num_repeat ${n_repeat} --gamma ${gamma} --tau_stim ${tau} --save_every_stim ${si} --start_epoch ${se} --epoch ${e} --set_wrp ${swrp} --wrp_path ${wrp_path} --r_in_path ${r_in_path} --theta_path ${theta_path} --out_path ${out_path} --postfix ${postfix}
EOF
done

# wait for all jobs to finish
wait