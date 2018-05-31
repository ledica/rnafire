#!/bin/ksh
nomeentrada="entrada_10_anos_serra_espinhaco_12_pontos_tsoi"

rm -rf maxmin_${nomeentrada}

nAmostra=$(wc -l ${nomeentrada}  | awk '{print $1}'  )
nCelula=$(wc -w ${nomeentrada} | awk '{print $1}'  )
nEntrada=$(( (${nCelula})/${nAmostra} ))
#ver cabecalho
var='data mes lat lon data/cbiol_10_.txt data/cbiol_11_.txt data/cbiol_12_.txt data/cbiol_1_.txt data/cbiol_2_.txt data/cbiol_3_.txt data/cbiol_4_.txt data/cbiol_5_.txt data/cbiol_6_.txt data/cbiol_7_.txt data/cbiol_8_.txt data/cbiol_9_.txt data/cbiow_10_.txt data/cbiow_11_.txt data/cbiow_12_.txt data/cbiow_1_.txt data/cbiow_2_.txt data/cbiow_3_.txt data/cbiow_4_.txt data/cbiow_5_.txt data/cbiow_6_.txt data/cbiow_7_.txt data/cbiow_8_.txt data/cbiow_9_.txt data/frac_10_.txt data/frac_11_.txt data/frac_12_.txt data/frac_1_.txt data/frac_2_.txt data/frac_3_.txt data/frac_4_.txt data/frac_5_.txt data/frac_6_.txt data/frac_7_.txt data/frac_8_.txt data/frac_9_.txt data/rain.txt data/totalit.txt data/totbio.txt data/ua.txt data/wsoi.txt modis'
arr=(${var// / })

i=1
export LC_ALL=C
while [ $i -le $nEntrada ]
do
    maior=$(xargs -n 1 <<< $(awk '{print $'${i}'}' ${nomeentrada} ) | sort -nr | head -1);
    menor=$(xargs -n 1 <<<  $(awk '{print $'${i}'}' ${nomeentrada} ) | sort -n | head -1);
    #if [ $maior != $menor ] ;
    #	then
    	echo ${arr[$((i-1))]}
    	print $i ${maior} ${menor}
    	
    #fi
    echo ${maior} ${menor} >> maxmin_${nomeentrada}
    i=$(( i+=1 ));
done

x=$( awk '{print $1}' maxmin_${nomeentrada} )
n=$( awk '{print $2}' maxmin_${nomeentrada} )
print ${x} > maxmin_${nomeentrada}
print ${n} >> maxmin_${nomeentrada}
exit

mkdir ${nomeentrada}