# -------------------------------------------------------------------------------
# Definition of functions used for COSMO output analysis
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# define directories
#
directories() {
    IFS="|"
    list_1h="SNOW_CON${IFS}SNOW_GSP${IFS}TOT_PREC${IFS}T_2M${IFS}QV_2M${IFS}PS"
    list_1h2="ASHFL_S${IFS}ALHFL_S${IFS}ASOB_S${IFS}ASWDIR_S${IFS}ASWDIFD_S${IFS}ATHB_S${IFS}ATHD_S${IFS}CAPE_ML${IFS}CIN_ML${IFS}AEVAP_S"
    list_3h="PMSL${IFS}CLCT${IFS}CLCL${IFS}CLCM${IFS}CLCH${IFS}T_G${IFS}U_10M${IFS}V_10M${IFS}ALB_RAD${IFS}ASOB_T${IFS}ASOD_T${IFS}ATHB_T${IFS}TQV${IFS}TQC${IFS}TQI${IFS}TWATER${IFS}TWATFLXU${IFS}TWATFLXV${IFS}HPBL${IFS}T_SO${IFS}TDIV_HUM"
    list_3h3D="FI${IFS}QV${IFS}T${IFS}U${IFS}V${IFS}W"
    list_24h3D="TADV_SUM${IFS}TCONV_SUM${IFS}TCONVLH_SUM${IFS}TTTUR_SUM${IFS}SOHR_SUM${IFS}THHR_SUM${IFS}TGSCP_SUM${IFS}TMPHYS_SUM"
    list_24h="VMAX_10M${IFS}H_SNOW${IFS}W_SNOW${IFS}TMAX_2M${IFS}TMIN_2M${IFS}DURSUN${IFS}RUNOFF_S${IFS}RUNOFF_G${IFS}SNOW_MELT${IFS}W_SO"

    pressure=(10000 20000 30000 40000 50000 60000 70000 85000 92500 100000)
    st1=("DJF" "MAM" "JJA" "SON")

    if [[ "${IFS}${list_1h[*]}${IFS}" =~ "${IFS}$1${IFS}" ]]; then
      subDir=1h
    elif [[ "${IFS}${list_1h2[*]}${IFS}" =~ "${IFS}$1${IFS}" ]]; then
       subDir=1h2
    elif [[ "${IFS}${list_3h[*]}${IFS}" =~ "${IFS}$1${IFS}" ]]; then
       subDir=3h
    elif [[ "${IFS}${list_3h3D[*]}${IFS}" =~ "${IFS}$1${IFS}" ]]; then
       subDir=3h3D
    elif [[ "${IFS}${list_24h3D[*]}${IFS}" =~ "${IFS}$1${IFS}" ]]; then
       subDir=24h3D
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

}

# -------------------------------------------------------------------------------
# merge ncfiles
#
mergeFiles() {
    directories $1 $2

    Dir=/store/c2sm/pr04/rxiang/data_lmp
    # Dir=/project/pr94/rxiang/data_lmp
    simname=$3
    year=$4
    inPath=$Dir/$4**_$3/lm_$reso/$subDir
    outPath=/project/pr133/rxiang/data/cosmo/EAS$2_$3/$subDir/$1
    dayPath=/project/pr133/rxiang/data/cosmo/EAS$2_$3/day/$1

    [ ! -d "$outPath" ] && mkdir -p "$outPath"
    [ ! -d "$dayPath" ] && mkdir -p "$dayPath"

    # if file exits, remove it
    if test -f "$outPath/$4_$1.nc"; then
        rm $outPath/$4_$1.nc
    fi

    cdo mergetime $inPath/$1.nc $outPath/$4_$1.nc
    echo "files $1 merged"

    if [[ "${IFS}${list_3h3D[*]}${IFS}" =~ "${IFS}$1${IFS}" ]]; then
      for p in "${pressure[@]}"
      do
        cdo -select,level=$p $outPath/$4_$1.nc $outPath/$4_$1_$p.nc
      done
      # rm $outPath/$4_$1.nc
    fi

    if [ "$1" == "TOT_PREC" ] || [ "$1" == "TQV" ]; then
      cdo -shifttime,-30minutes $outPath/$4_$1.nc $outPath/$4_$1_sft.nc
      cdo daysum $outPath/$4_$1_sft.nc $dayPath/$4_$1.nc
      rm $outPath/$4_$1_sft.nc
      echo "shift time"
    else
      echo "no shift time"
    fi
}

# -------------------------------------------------------------------------------
# compute seasonalities
#
seasonal() {
    directories $1 $2

    echo "compute seasonalities"

    inPath=/project/pr133/rxiang/data/cosmo/EAS$2_$3/$subDir/$1
    outPath=/project/pr133/rxiang/data/cosmo/EAS$2_$3/szn/$1

    [ ! -d "$outPath" ] && mkdir -p "$outPath"

    for s in "${st1[@]}"
    do 
      if [[ "${IFS}${list_3h3D[*]}${IFS}" =~ "${IFS}$1${IFS}" ]]; then
      echo "true"
        for p in "${pressure[@]}"
        do
          cdo -select,season="$s" $inPath/$4_$1_${p}.nc $inPath/$4_$1_${p}_TS_${s}.nc
          cdo timmean $inPath/$4_$1_${p}_TS_${s}.nc $outPath/$4_$1_${p}_${s}.nc
          rm $inPath/$4_$1_${p}_TS_${s}.nc
        done
      else
      cdo -select,season="${s}" $inPath/$4_$1.nc $inPath/$4_$1_TS_${s}.nc
      cdo timmean $inPath/$4_$1_TS_${s}.nc $outPath/$4_$1_${s}.nc
      rm $inPath/$4_$1_TS_${s}.nc
      fi
    done
}

# -------------------------------------------------------------------------------
# compute vertically integrated water vapor transport
#
IVT() {
    echo "compute $1"

    inPath=/project/pr94/rxiang/analysis/EAS$2_$3
    outPathU=/project/pr94/rxiang/analysis/EAS$2_$3/IVT_U
    outPathV=/project/pr94/rxiang/analysis/EAS$2_$3/IVT_V

    [ ! -d "$outPathU" ] && mkdir -p "$outPathU"
    [ ! -d "$outPathV" ] && mkdir -p "$outPathV"

    cdo -L -expr,'qvu=U*QV' -merge $inPath/U/$4_U.nc $inPath/QV/$4_QV.nc $outPathU/$4_QVU.nc
    cdo -L -expr,'qvv=V*QV' -merge $inPath/V/$4_V.nc $inPath/QV/$4_QV.nc $outPathV/$4_QVV.nc
    ncwa -N -v qvu -w pressure -a pressure $outPathU/$4_QVU.nc $outPathU/$4_IVT_U.nc
    ncwa -N -v qvv -w pressure -a pressure $outPathV/$4_QVV.nc $outPathV/$4_IVT_V.nc
}
