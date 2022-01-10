#!/bin/bash
source cosmoFunctions.sh

#-------------------------------------------------------------------------------
# 
declare -a var=("TOT_PREC" "T_2M") 
resolution=11
sim=ctrl
year=01

for i in "${var[@]}"
do
  mergeFiles $i $resolution $sim $year
  seasonal $i $resolution $sim $year
done
