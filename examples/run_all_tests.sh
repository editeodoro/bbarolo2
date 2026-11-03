#!/bin/bash
# Pure-Python equivalent of examples/test/testall.sh.
# Run it from the BBarolo main directory (the one containing examples/):
#     cd /path/to/BBarolo && bash pyBBarolo_pure/examples/run_all_tests.sh
# Outputs go to ./output_py/<test>/ and logs to ./output_py/<test>.log
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BB="$HERE/bin/BBarolo"
mkdir -p output_py
for t in 3dfit 2dfit ellprof galmod moment rend3d search smooth smoothspec spacepar; do
    start=$(date +%s)
    $BB -p examples/test/n2403_$t.par OUTFOLDER=./output_py/n2403_$t/ > ./output_py/n2403_$t.log 2>&1
    echo "n2403_$t: exit code $?, $(( $(date +%s) - start )) s"
done
