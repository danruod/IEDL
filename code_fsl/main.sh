#!/bin/bash

#SBATCH --job-name=IEDL
#SBATCH --output=/research/d6/gds/drdeng/ood/IEDL-main/IEDL-FSL/2r.txt

CFGPREFIXLIST=("1_mini/5w-edl" / 
              "1_mini/5w-iedl")

for CFGPREFIX in "${CFGPREFIXLIST[@]}"; do
  echo "Running Configuration $CFGPREFIX"
  echo "  ==> The json config will be read from ./configs/${CFGPREFIX}.json"
  echo "  ==> The results csv file will be saved at ./results/${CFGPREFIX}.csv"
  echo "  + python main.py --configid ${CFGPREFIX}"
  python main.py --configid ${CFGPREFIX} --suffix 'final'
  echo "----------------------------------------"
done