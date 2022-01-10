#!/bin/bash
source cosmoFunctions.sh

#-------------------------------------------------------------------------------
# 
declare -a var=("TOT_PREC" "T_2M") 
declare -a sim=("ctrl" "topo1")
resolution=11
year=01

for i in "${var[@]}"
do
  for j in "${sim[@]}"
  do
    mergeFiles $i $resolution $j $year
    seasonal $i $resolution $j $year
done
