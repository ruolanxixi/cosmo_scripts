#!/bin/bash
source cosmoFunctions.sh

#-------------------------------------------------------------------------------
# 
# TOT_PREC U V --- summer precipitation and summer winds at 850 hPa [kg m-2 >> mm day-1]
# W TQV --- summer vertical velocity at 500 hPa and summer TQV (total precipitable water) [kg m-2 >> mm day-1]
# T FI --- summer temperature and geopotential at 500 hPa [K >> oC]
# SOHR_RAD THHR_RAD DT_CON DT_SSO --- diabatic heating [K s-1 >> K day-1]
# Omega --- lagrangian tendency of air pressure [Pa s-1 >> hPa day-1]
# PMSL --- winter monsoon winds at 850 hPa and see level pressure [Pa >> hPa]

# declare -a var=("TOT_PREC" "U" "V" "W" "TQV" "QV" "T_2M" "T" "FI" "SOHR_RAD" "THHR_RAD" "DT_CON" "DT_SSO" "OMEGA" "PMSL") 
declare -a var=("PMSL")
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
