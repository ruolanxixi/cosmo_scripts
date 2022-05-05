#!/bin/bash
source cosmoFunctions.sh

# -------------------------------------------------------------------------------
#

declare -a sim=("ctrl" "topo1")
resolution=11
year=01

for c in "${sim[@]}"
  do
    IVT ivt $resolution $c $year
  done
