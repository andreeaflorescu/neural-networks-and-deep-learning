#/bin/bash
#module load libraries/openmpi-1.6-gcc-4.4.6
#export PYTHONPATH=/usr/lib64/python2.6/site-packages/openmpi/mpi4py
#mpich-autoload
module load openmpi-x86_64
echo "Processes $1"
mpirun -n $1 python main.py
