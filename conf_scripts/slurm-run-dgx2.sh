#!/usr/bin/env bash
#SBATCH --partition=dgx-2v_32g
#SBATCH --nodelist=dgx2-03
#SBATCH --time=1440

echo "Run time: `date`"
#srun -N 1 -p dgx-1v_16g /home/shaohuais/repos/dsr/finetune.sh
#source /home/shaohuais/.bashrc
#source /home/shaohuais/tf1.13.1/bin/activate
/home/shaohuais/copydata.sh
sleep 300
/home/shaohuais/repos/p2p-topk/batch.sh
