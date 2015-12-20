#!/bin/bash
#for i in {1..18}
#do
	#qsub -q ibm-nehalem.q -cwd -pe openmpi*1 $1 run_ann.sh $1
	qsub -cwd -pe openmpi*1 $1 run_ann.sh $1
#done	
