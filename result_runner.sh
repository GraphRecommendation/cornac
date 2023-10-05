#!/bin/bash
DATASETS="cellphone toy"
#METHODS="trirank bpr narre narre-bpr hrdr hrdr-bpr ngcf kgat lightgcn lightrla hypar"
METHODS="globalrla-le"
ABLATION="{'globalrla-lg': {'fixed': {'preference_module':'lightgcn'}, 'ablation': ['graph_type', ['ao', 'a', 'o', 'as', 'os', 'n']]},
'globalrla-le': {'fixed': {'preference_module':'mf','learn_explainability':True}, 'ablation': ['learn_weight', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 5.0]]}}"
source .venv/bin/activate
for DATASET in $DATASETS; do
  echo $DATASET;
  python selection_and_results.py results $DATASET $METHODS  --method_ablation_dict "${ABLATION}";
done