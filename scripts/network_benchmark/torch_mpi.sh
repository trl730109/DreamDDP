nworkers="${nworkers:-2}"
MPIPATH=/usr/local/openmpi/openmpi-4.0.1
PY=/usr/local/bin/python
$MPIPATH/bin/mpirun --prefix $MPIPATH -np $nworkers -hostfile cluster$nworkers -bind-to none -map-by slot \
        $PY ./torch_allreduce.py 
