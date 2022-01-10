#-------------------------------------------------------------------------------
# Definition of functions used for COSMO output analysis
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# define directories
#
IFS="|"
list_1h="SNOW_CON${IFS}SNOW_GSP${IFS}RAIN_CON${IFS}RAIN_GSP${IFS}TOT_PREC${IFS}TD_2M${IFS}T_2M${IFS}U_10M${IFS}V_10M${IFS}CLCT${IFS}CLCL${IFS}CLCM${IFS}CLCH"
list_3h="ALHFL_S${IFS}ATHD_S${IFS}ATHU_S${IFS}ASHFL_S${IFS}ASOD_T${IFS}ASOB_T${IFS}ASOB_S${IFS}ATHB_T${IFS}ATHB_S${IFS}ASWDIFD_S${IFS}ASWDIFU_S${IFS}ASWDIR_S${IFS}ASOBC_S${IFS}ASOBC_T${IFS}ATHBC_S${IFS}ATHBC_T${IFS}DURSUN${IFS}PMSL${IFS}PS${IFS}QV_2M${IFS}RELHUM_2M${IFS}ALB_RAD${IFS}AEVAP_S"
list_6h="H_SNOW${IFS}RUNOFF_G${IFS}RUNOFF_S${IFS}TQC${IFS}TQI${IFS}TQV${IFS}TQR${IFS}TQS${IFS}TQG${IFS}HPBL${IFS}SNOW_MELT${IFS}WTDEPTH"
list_6h3D="FI${IFS}QV${IFS}T${IFS}U${IFS}V"
list_24h="TMAX_2M${IFS}TMIN_2M${IFS}AER_SO4${IFS}AER_DUST${IFS}AER_BC"

pressure=(10000 20000 30000 40000 50000 60000 70000 85000 92500)
st1=("DJF" "MAM" "JJA" "SON")
#-------------------------------------------------------------------------------
# merge ncfiles
#
mergeFiles() {

if [[ "${IFS}${list_1h[*]}${IFS}" =~ "${IFS}$1${IFS}" ]]; then
  subDir=1h
elif [[ "${IFS}${list_3h[*]}${IFS}" =~ "${IFS}$1${IFS}" ]]; then
   subDir=3h
elif [[ "${IFS}${list_6h[*]}${IFS}" =~ "${IFS}$1${IFS}" ]]; then
   subDir=6h
elif [[ "${IFS}${list_6h3D[*]}${IFS}" =~ "${IFS}$1${IFS}" ]]; then
   subDir=6h3D
elif [[ "${IFS}${list_24h[*]}${IFS}" =~ "${IFS}$1${IFS}" ]]; then
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
outPath=/project/pr94/rxiang/analysis/EAS$2_$3/$1

[ ! -d "$outPath" ] && mkdir -p "$outPath"

# if file exits, remove it
if test -f "$outPath/$4_$1_mergetime.nc"; then
    rm $outPath/$4_$1_mergetime.nc
fi

cdo mergetime $inPath/$1.nc $outPath/$4_$1_mergetime.nc
echo "files merged"

pressure=(10000 20000 30000 40000 50000 60000 70000 85000 92500)
if [[ "${IFS}${list_6h3D[*]}${IFS}" =~ "${IFS}$1${IFS}" ]]; then
  for p in "${pressure[@]}"
  do
    cdo -select,level=$p $outPath/$4_$1_mergetime.nc $outPath/$4_$1_mergetime_$p.nc
  done
  rm $outPath/$4_$1_mergetime.nc
fi

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
# compute horizontal wind
#
horizontal() {
echo "compute horizontal wind"
echo "start $1"
uPath=/project/pr94/rxiang/analysis/EAS$2_$3/u
vPath=/project/pr94/rxiang/analysis/EAS$2_$3/v
outPath=/project/pr94/rxiang/analysis/EAS$2_$3/wind

[ ! -d "$outPath" ] && mkdir -p "$outPath"
for s in "${st1[@]}"
do 
  for p in "${pressure[@]}"
  do
  cdo sqrt -add -sqr $uPath/$4_U_mergetime_$p_TS_${s}.nc -sqr $vPath/$4_V_mergetime_$p_TS_${s}.nc outPath/$4_wind_mergetime_$p_TS_${s}.nc
  cdo timmean outPath/$4_wind_mergetime_$p_TS_${s}.nc outPath/$4_wind_mergetime_$p_${s}.nc
  rm outPath/$4_wind_mergetime_$p_TS_${s}.nc
  done
done
}

#-------------------------------------------------------------------------------
# compute seasonalities
#
seasonal() {
echo "compute seasonalities"

Path=/project/pr94/rxiang/analysis/EAS$2_$3/$1

[ ! -d "$Path" ] && mkdir -p "$Path"

for s in "${st1[@]}"
do 
  if [[ "${IFS}${list_6h3D[*]}${IFS}" =~ "${IFS}$1${IFS}" ]]; then
  echo "true"
    for p in "${pressure[@]}"
    do
      cdo -select,season="$s" $Path/$4_$1_mergetime_$p.nc $Path/$4_$1_mergetime_${p}_TS_$s.nc
      if [ "$1" != "U" ] && [ "$1" != "V" ]; then
        cdo timmean $Path/$4_$1_mergetime_${p}_TS_$s.nc $Path/$4_$1_mergetime_${p}_$s.nc
        rm $Path/$4_$1_mergetime_${p}_TS_$s.nc
      fi
    done
  else
    cdo -select,season="$s" $Path/$4_$1_mergetime.nc $Path/$4_$1_mergetime_TS_$s.nc
    cdo timmean $Path/$4_$1_mergetime_TS_$s.nc $Path/$4_$1_mergetime_$s.nc
    rm $Path/$4_$1_mergetime_TS_$s.nc
  fi
done
}
