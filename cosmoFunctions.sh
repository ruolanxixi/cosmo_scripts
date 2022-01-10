#-------------------------------------------------------------------------------
# Definition of functions used for COSMO output analysis
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# merge ncfiles
#
mergeFiles() {

list_1h="SNOW_CON SNOW_GSP RAIN_CON RAIN_GSP TOT_PREC TD_2M T_2M U_10M V_10M CLCT CLCL CLCM CLCH"
list_3h="ALHFL_S ATHD_S ATHU_S ASHFL_S ASOD_T ASOB_T ASOB_S ATHB_T ATHB_S ASWDIFD_S ASWDIFU_S ASWDIR_S ASOBC_S ASOBC_T ATHBC_S ATHBC_T DURSUN PMSL PS QV_2M RELHUM_2M ALB_RAD AEVAP_S"
list_6h="H_SNOW RUNOFF_G RUNOFF_S TQC TQI TQV TQR TQS TQG HPBL SNOW_MELT WTDEPTH"
list_6h3D="FI QV T U V"
list_24h="TMAX_2M TMIN_2M AER_SO4 AER_DUST AER_BC"

if [[ "$list_1h[*]}" =~ "$1" ]]; then
  subDir=1h
elif [[ "$list_3h[*]}" =~ "$1" ]]; then
   subDir=3h
elif [[ "$list_6h[*]}" =~ "$1" ]]; then
   subDir=6h
elif [[ "$list_6h3D[*]}" =~ "$1" ]]; then
   subDir=6h3D
elif [[ "$list_24h[*]}" =~ "$1" ]]; then
   subDir=24h
else
   echo "variable $1 doesn't exit in cosmo output"
fi

if [ "$2" == "11" ]; then
  reso=coarse
elif [ "$2" == "44" ]; then
  reso=fine
fi


Dir=/project/pr94/rxiang/data_lmp
simname=$3
year=$4
inPath=$Dir/$4**_$3/lm_$reso/$subDir
outPath=/project/pr94/rxiang/analysis/EAS$2_$3

# if file exits, remove it
if test -f "$outPath/$4_$1_mergetime.nc"; then
    rm $outPath/$4_$1_mergetime.nc
fi

ls $inPath
cdo mergetime $inPath/$1.nc $outPath/$4_$1_mergetime.nc
echo "files merged"

if [ "$1" == "TOT_PREC" ]; then
  cdo -shifttime,-30minutes $outPath/$4_$1_mergetime.nc $outPath/$4_$1_mergetime_sft.nc
  rm $outPath/$4_$1_mergetime.nc
  mv $outPath/$4_$1_mergetime_sft.nc $outPath/$4_$1_mergetime.nc
  echo "shift time"
  else
  echo "no shift time"
fi
}

#-------------------------------------------------------------------------------
# compute seasonalities
#
seasonal() {
st1=("DJF" "MAM" "JJA" "SON")
inPath=/project/pr94/rxiang/analysis/EAS$2_$3
outPath=/project/pr94/rxiang/analysis/EAS$2_$3/seasonal

for i in "${st1[@]}"
do
  cdo -select,season="$i"  $4_$1_mergetime.nc $outPath/$4_$1_mergetime_TS_${i}.nc
  cdo timmean $outPath/$4_$1_mergetime_TS_${i}.nc $outPath/$4_$1_mergetime_${i}.nc
  rm $outPath/$4_$1_mergetime_TS_${i}.nc
done
}
