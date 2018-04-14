#!/bin/bash
for j in `seq 1 3`; do
    for i in `seq 1 9`; do
        for sched in piecewise-constant clr-cosine clr-triangle clr-step; do
           lr=${i}e-${j}
           echo ${sched}-${lr}
           tox -e python -- run_single_experiment.py -l ${lr} -d all -y basic_classifier_MNIST.yaml -r ${sched}
        done
    done
done

for lr in 1.0 1.1 1.2 1.3 1.4 1.5 1.6; do
    for sched in piecewise-constant clr-cosine clr-triangle clr-step; do
       echo ${sched}-${lr}
       tox -e python -- run_single_experiment.py -l ${lr} -d all -y basic_classifier_MNIST.yaml -r ${sched}
    done
done
