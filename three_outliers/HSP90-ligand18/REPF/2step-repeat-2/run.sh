#!/bin/bash
#TOOLS_PATH=/HOME/scz1641/run/rest_contrast/MCL-1/Alchemd
TOOLS_PATH=../../Alchemd
#conda activate openmm-plumed
if [ -e MD_Finish ] ; then
    echo 'MD_Finish exist! Would not run simulation!'
else  
    python $TOOLS_PATH/openmm-FEP-run.py -p protein.prmtop -c protein.rst7 -i input.txt > mdout 2>&1
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        touch MD_Finish
    fi
fi
