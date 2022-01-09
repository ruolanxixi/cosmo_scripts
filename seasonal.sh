#!/bin/bash
source cosmoFunctions.sh

#-------------------------------------------------------------------------------
# 
var=TOT_PREC
resolution=11
sim=topo1
year=01
mergeFiles $var $resolution $sim $year
