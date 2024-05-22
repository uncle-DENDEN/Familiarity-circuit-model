#!/bin/bash

# define seeds and directories
taus=(500)
sis=(250)
gammas=(1)

noises=(0)
n_trials=(1)

gammas_tr=(30)
swrps=(0)
# wrp_path='.'
r_in_paths=("tr_berkeley/r_in_abs_500_64_5_5.npy") 
postfixs=("v3")
out_paths=("Response_scale_v3")

# loop through each seed and directory pair
for i in {0..0}; do
  for j in {0..0}; do
    tau=${taus[0]}
    save_every_stim=${sis[0]}
    gamma=${gammas[0]}

    noise=${noises[0]}
    n_trial=${n_trials[0]}
    
    gamma_tr=${gammas_tr[0]}
    swrp=${swrps[j]}
    if [ $j -eq 0 ]; then
        wrp_path="."
    elif [ $j -eq 1 ]; then
        wrp_path="."
    elif [ $j -eq 2 ]; then
        wrp_path="."
    fi

    r_in_path=${r_in_paths[0]}
    postfix=${postfixs[j]}
    out_path=${out_paths[0]}

    logfile=/user_data/weifanw/familiarity/runtest-$i-$j.txt

    echo "Running test $i: ${postfix} with ckpt ${wrp_path}"

    srun -p gpu --gpus=1 --mem-per-gpu=12GB --mem=80GB --time=8:00:00 --pty -n1 bash << EOF &> $logfile &
      module load anaconda3
      source activate fam
      cd /user_data/weifanw/familiarity
      python test.py --tau_stim ${tau} --save_every_stim ${save_every_stim} --gamma ${gamma} --noise ${noise} --n_trials ${n_trial} --set_wrp ${swrp} --wrp_path ${wrp_path} --postfix ${postfix} --r_in_path ${r_in_path} --out_path ${out_path}
EOF
  done
done

# wait for all jobs to finish
wait