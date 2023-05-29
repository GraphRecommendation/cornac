#!/bin/bash

source .venv/bin/activate
export PYTHONPATH=${PWD}

METHODS='globalrla-e' # hrdr narre'
DATASETS='cellphone' # toy computer camera'
MATCHES='a ao'
METHODOLOGIES='greedy_item greedy_user weighted item'
for DATASET in $DATASETS; do
  for METHOD in $METHODS; do
    if [[ "$METHOD" =~ ^globalrla.* ]]; then
      for MM in $MATCHES; do
        for METHO in $METHODOLOGIES; do
          echo $DATASET $METHOD $MM $METHO
          python statistics/method_graph_overlap.py "results" $DATASET $METHOD --method_kwargs "{ 'globalrla':{'methodology':'${METHO}','weighting':'attention'} ,'globalrla-e':{'methodology':'${METHO}','weighting':'LightRLAaos'}, 'narre':{}, 'hrdr':{}, 'matching_method':'${MM}'}"
        done
      done
    else
      echo $DATASET $METHOD
      python statistics/method_graph_overlap.py "results" $DATASET $METHOD --method_kwargs "{ 'globalrla':{'methodology':'item','weighting':'attention'}, 'narre':{}, 'hrdr':{}, 'matching_method':'ao'}"
    fi
  done
done