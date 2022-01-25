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
    seasonal IVT_U $resolution $c $year
    seasonal IVT_V $resolution $c $year
  done
