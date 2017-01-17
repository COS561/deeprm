#!/bin/bash
for AMP in 2 3 4 5 6 7
do
	echo $AMP
	let AMP2=$AMP\*2
	echo $AMP2
	python launcher.py --amplitude="$AMP2" --experiment_name="parallel_$AMP2" --exp_type=test --simu_len=50 --num_ex=10 --rnn=True --pg_re=nets/period_amp_1850.pkl --unseen=True
done
