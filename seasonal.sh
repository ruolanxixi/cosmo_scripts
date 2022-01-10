#!/bin/bash
source cosmoFunctions.sh

#-------------------------------------------------------------------------------
# 
declare -a var=("TOT_PREC" "T_2M" "T" "TQV" "U" "V") 
declare -a sim=("ctrl" "topo1")
resolution=11
year=01

for v in "${var[@]}"
do
  for c in "${sim[@]}"
  do
    mergeFiles $v $resolution $c $year
    seasonal $v $resolution $c $year
  done
done
