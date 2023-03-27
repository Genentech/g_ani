#!/bin/csh -f

set n=4667.sdf
source  /gstore/apps/Lmod/default/lmod/lmod/init/csh
ml cdd/python/dev
conda activate py36
ml nvidia/driver

mysub.py -limit 20:00 -nGPU 1 -totalMem 30 -nCPU 2 -jobName $n:r.strian <<coms
#!/bin/tcsh -f
   time sdfNNPConfAnalysis.pl -in $n -out $n:r.out.sdf -sampleOtherMin
coms

