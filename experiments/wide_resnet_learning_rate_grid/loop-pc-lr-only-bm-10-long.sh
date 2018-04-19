#!/bin/bash
for j in 2 1 0; do
    for i in `seq 1 9`; do
       lr=${i}e-${j}
       echo "Learning Rate: ${lr}"
       tox -e python -- run_single_experiment.py \
           --max-learning-rate ${lr} \
           --min-learning-rate ${lr} \
           --experiment-directory pc-lr-only \
           --rate-schedule piecewise-constant-long \
           --batch-multiplier 10
    done
done
