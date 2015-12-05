#!/bin/bash
for i in {1..18}
do
	echo "Number of processes: $i"
	mpirun -np $i python main.py > results.txt
done	
