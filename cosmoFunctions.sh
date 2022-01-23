# -------------------------------------------------------------------------------
# Definition of functions used for COSMO output analysis
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# define directories
#
IFS="|"
list_1h="SNOW_CON${IFS}SNOW_GSP${IFS}RAIN_CON${IFS}RAIN_GSP${IFS}TOT_PREC${IFS}TD_2M\
${IFS}T_2M${IFS}U_10M${IFS}V_10M${IFS}CLCT${IFS}CLCL${IFS}CLCM${IFS}CLCH"
list_3h="ALHFL_S${IFS}ATHD_S${IFS}ATHU_S${IFS}ASHFL_S${IFS}ASOD_T${IFS}ASOB_T${IFS}\
ASOB_S${IFS}ATHB_T${IFS}ATHB_S${IFS}ASWDIFD_S${IFS}ASWDIFU_S${IFS}ASWDIR_S${IFS}ASOBC_S\
${IFS}ASOBC_T${IFS}ATHBC_S${IFS}ATHBC_T${IFS}DURSUN${IFS}PMSL${IFS}PS${IFS}QV_2M${IFS}RELHUM_2M${IFS}ALB_RAD${IFS}AEVAP_S"
list_6h="H_SNOW${IFS}RUNOFF_G${IFS}RUNOFF_S${IFS}TQC${IFS}TQI${IFS}TQV${IFS}TQR${IFS}\
TQS${IFS}TQG${IFS}HPBL${IFS}SNOW_MELT${IFS}WTDEPTH"
list_6h3D="FI${IFS}QV${IFS}T${IFS}U${IFS}V${IFS}W${IFS}SOHR_RAD${IFS}THHR_RAD${IFS}DT_CON${IFS}DT_SSO${IFS}OMEGA${IFS}"
list_24h="TMAX_2M${IFS}TMIN_2M${IFS}TWATER${IFS}TWATFLXU${IFS}TWATFLXV${IFS}T_SO${IFS}W_SO${IFS}VMAX_10M${IFS}"

pressure=(10000 20000 30000 40000 50000 60000 70000 85000 92500)
st1=("DJF" "MAM" "JJA" "SON")
# -------------------------------------------------------------------------------
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

cdo mergetime $inPath/$1.nc $outPath/$4_$1.nc
echo "files merged"

pressure=(10000 20000 30000 40000 50000 60000 70000 85000 92500)
if [[ "${IFS}${list_6h3D[*]}${IFS}" =~ "${IFS}$1${IFS}" ]]; then
  for p in "${pressure[@]}"
  do
    cdo -select,level=$p $outPath/$4_$1.nc $outPath/$4_$1_$p.nc
  done
  # rm $outPath/$4_$1.nc
fi

if [ "$1" == "TOT_PREC" ] || [ "$1" == "TQV" ]; then
  cdo -shifttime,-30minutes $outPath/$4_$1.nc $outPath/$4_$1_sft.nc
  cdo daysum $outPath/$4_$1_sft.nc $outPath/$4_$1_daysum.nc
  rm $outPath/$4_$1.nc $outPath/$4_$1_sft.nc
  mv $outPath/$4_$1_daysum.nc $outPath/$4_$1.nc
  echo "shift time"
else
  echo "no shift time"
fi
}

# -------------------------------------------------------------------------------
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
      cdo -select,season="$s" $Path/$4_$1_${p}.nc $Path/$4_$1_${p}_TS_${s}.nc
      cdo timmean $Path/$4_$1_${p}_TS_${s}.nc $Path/$4_$1_${p}_${s}.nc
      rm $Path/$4_$1_${p}_TS_${s}.nc
    done
  else
  cdo -select,season="${s}" $Path/$4_$1.nc $Path/$4_$1_TS_${s}.nc
  cdo timmean $Path/$4_$1_TS_${s}.nc $Path/$4_$1_${s}.nc
  rm $Path/$4_$1_TS_${s}.nc
  fi
done
}

# -------------------------------------------------------------------------------
# compute vertically integrated water vapor transport
#
IVT() {
echo "compute IVT"

inpath=/project/pr94/rxiang/analysis/EAS$2_$3/
outPath=/project/pr94/rxiang/analysis/EAS$2_$3/IVT

[ ! -d "$outPath" ] && mkdir -p "$outPath"

cdo -expr,'qvu=U*QV' -merge $inPath/U/$4_U.nc $inPath/QV/$4_QV.nc $outpath/$4_QVU.nc
cdo -expr,'qvv=V*QV' -merge $inPath/V/$4_V.nc $inPath/QV/$4_QV.nc $outpath/$4_QVV.nc
ncwa -N -v qvu -w pressure -a pressure $outpath/$4_QVU.nc outPath/$4_IVT_U.nc
ncwa -N -v qvv -w pressure -a pressure $outpath/$4_QVV.nc outPath/$4_IVT_V.nc
}
