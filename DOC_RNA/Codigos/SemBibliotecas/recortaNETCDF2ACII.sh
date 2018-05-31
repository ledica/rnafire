#!/bin/ksh


dir1='/home/leticia/CCST/REDES_INLAND_2018/INLAND_BOX/dados/inland/inland_rna/data/exp5_cerrado/output/monthly'
dir2='/home/leticia/CCST/REDES_INLAND_2018/INLAND_BOX/dados/inland/inland_rna/data/exp5_cerrado/output/monthly-pft'
dir3='/home/leticia/CCST/REDES_INLAND_2018/INLAND_BOX/dados/inland/inland_rna/data/exp5_cerrado/output/mcd14ml_clim_netcdf'
var1='wsoi totalit ua rain tsoi'
var2='cbiol cbiow frac'

rm -rf data
mkdir data

cdo -cat ${dir1}/*-21??.nc data/monthly-2100-2199.nc

cdo griddes data/monthly-2100-2199.nc > gride

cdo -sellonlatbox,-45,-43,-18.5,-17 -setmissval,0 data/monthly-2100-2199.nc data/monthly.nc

cdo -setmissval,0 data/monthly-2100-2199.nc data/monthly.nc

cdo setrtoc2,0,10000000000000,1,0 -seltimestep,1,1 -selvar,ua data/monthly-2100-2199.nc data/base.nc
rm data/monthly-2100-2199.nc



	rm data/monthly-pft-2100-2199.nc
	cdo -cat ${dir2}/*21??.nc data/temp.nc

for e in ${var2};
do
	cdo selvar,${e} -setmissval,0 -sellonlatbox,-45,-43,-18.5,-17  data/temp.nc data/${e}.nc

	rm data/${e}.txt
	for i in {1..12}
	do
		cdo -sellevel,${i} data/${e}.nc data/${e}_${i}_.nc
	    cdo -outputtab,name,date,lat,lon,value data/${e}_${i}_.nc > data/${e}_${i}_.txt;

	done
	cdo enssum data/${e}_*_.nc data/tot_${e}.nc

done
    cdo enssum data/tot_cbiow.nc data/tot_cbiol.nc data/totbio.nc
    cdo -outputtab,name,date,lat,lon,value -setname,totbio data/totbio.nc > data/totbio.txt;

for e in ${var1};
do
	cdo selvar,${e} data/monthly.nc data/${e}.nc
	cdo -outputtab,name,date,lat,lon,value  data/${e}.nc > data/${e}.txt;
done

cdo settaxis,0000-01-01,12:00,1mon  ${dir3}/mcd14ml_clim_01.nc  data/mcd14ml_clim_01.nc
cdo settaxis,0000-02-01,12:00,1mon  ${dir3}/mcd14ml_clim_02.nc  data/mcd14ml_clim_02.nc
cdo settaxis,0000-03-01,12:00,1mon  ${dir3}/mcd14ml_clim_03.nc  data/mcd14ml_clim_03.nc
cdo settaxis,0000-04-01,12:00,1mon  ${dir3}/mcd14ml_clim_04.nc  data/mcd14ml_clim_04.nc
cdo settaxis,0000-05-01,12:00,1mon  ${dir3}/mcd14ml_clim_05.nc  data/mcd14ml_clim_05.nc
cdo settaxis,0000-06-01,12:00,1mon  ${dir3}/mcd14ml_clim_06.nc  data/mcd14ml_clim_06.nc
cdo settaxis,0000-07-01,12:00,1mon  ${dir3}/mcd14ml_clim_07.nc  data/mcd14ml_clim_07.nc
cdo settaxis,0000-08-01,12:00,1mon  ${dir3}/mcd14ml_clim_08.nc  data/mcd14ml_clim_08.nc
cdo settaxis,0000-09-01,12:00,1mon  ${dir3}/mcd14ml_clim_09.nc  data/mcd14ml_clim_09.nc
cdo settaxis,0000-10-01,12:00,1mon  ${dir3}/mcd14ml_clim_10.nc  data/mcd14ml_clim_10.nc
cdo settaxis,0000-11-01,12:00,1mon  ${dir3}/mcd14ml_clim_11.nc  data/mcd14ml_clim_11.nc
cdo settaxis,0000-12-01,12:00,1mon  ${dir3}/mcd14ml_clim_12.nc  data/mcd14ml_clim_12.nc
cdo -cat data/mcd14ml_clim_??.nc data/mcd14ml_clim.nc
rm data/mcd14ml_clim_??.nc

cdo -setmissval,-1 data/mcd14ml_clim.nc data/saidas_temp.nc
cdo remapcon,gride data/saidas_temp.nc data/saidas_temp2.nc
cdo -setmissval,-1 -mul data/base.nc data/saidas_temp2.nc data/saidas.nc

cdo -setmissval,-1 -sellonlatbox,-45,-43,-18.5,-17   -mul data/base.nc data/saidas_temp2.nc data/saidas.nc
cdo -outputtab,name,date,lat,lon,value data/saidas.nc > data/saidas.data;

#rm -rf *nc

exit