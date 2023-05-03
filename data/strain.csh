#!/bin/csh -f

set n=4667.sdf

mysub.py -limit 20:00 -nGPU 1 -totalMem 30 -nCPU 2 -jobName $n:r.strian <<coms
#!/bin/tcsh -f
   time sdfNNPConfAnalysis.pl -in $n -out $n:r.out.sdf -sampleOtherMin
coms

