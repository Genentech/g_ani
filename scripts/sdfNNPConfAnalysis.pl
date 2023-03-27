#!/usr/bin/env perl
use warnings;

$test = <<'TEST';

sdfNNPConfAnalysis.pl -in ~/dev/ml_qm_data/4667.sdf -out t.sdf -omegaOpt "-includeInput true -sampleHydrogens -searchff mmff94s -ewindow 50 -maxconfs 50 "

or

mysub.py -limit 0:20 -totalMem 15 -nCPU 2 -nGPU 1 -jobName testNNPConfA <<'coms'
#!/bin/tcsh -f
   source /system/gredit/clientos/etc/profile.d/lmod.csh;
   source ~cdduser/prd/bin/strain.nnp.csh
   time sdfNNPConfAnalysis.pl -in ~/dev/ml_qm_data/4667.sdf -out 4667.nnpconfa.sdf -sampleOtherMin -debug
'coms'
TEST


$script=$0;
$script =~ /^\// || ( $script="$ENV{PWD}/$script" );

use File::Copy;
use Getopt::Long;
use POSIX ":sys_wait_h";

#changes due to experiments with macrocycle
my($omegaOpt ) = "-includeInput true -sampleHydrogens -searchff mmff94s -ewindow 50 -maxconfs 650 ";
#my($omegaOpt ) = "-includeInput true -searchff mmff94s -ewindow 50 -maxconfs 10 ";
my($NNPCommand) = "sdfMOptimizer.py";
#my($NNPCommand) = "sdfGeomeTRIC.py";
my($nnpOpt) = "-maxiter 2000 -nGPU 1 -trust 1";
my($nnpConf) = "~smdi/prd/ml_qm_data/nnp/ani22/bb/bb.3.json";

$use = <<USE;
sdfNNPConfAnalysis.pl [-ref alignRef.sdf] [-nnpOpt s] [-omegaOpt s]
                      -in in.sdf -out out.sdf [-bad bad.sdf] [-sampleOtherMin]
                     [-outputBest maxRMSD] [-gMinConfSDF sdf]
  -ref  .............. align conformers to given 3d sss fragment
  -nnpConf ........... NNP .json file or directory for ANI
  -nnpOpt ............ additional option for the NNP. default "$nnpOpt"
                       local optimization will allways use -conj
  -omegaOpt .......... additional option for omega. def "$omegaOpt"
  -sampleOtherMin .... also look for other low energy low rmsd minima
  -failOnMultiInput .. Fail if more than on input record. (workaround for moe bug)
  -bad ............... if given compounds with problems will go to bad.sdf
  -outputBest ........ Output lowest energy pose with rmsd <= maxRMSD
  -gMinConfSDF ....... Output optimized conformations from conformational search to file.
  -nGPU .............. If 0 no GPU's are used if not given or > 0 and no GPU avaialble a GPU submission will be done
  -debug ............. print commands before executing

If used with sdfMultiplexer use the -groupByAtomCount to make sure the records
for one input structre are kept together.
USE

my(@ARG) = @ARGV;

my( $com, $in, $ref, $out, $tmpBase, $debug, $sampleOtherMin,
    $harmonic, $failOnMultiInput, $maxRMSD, $gMinConfSDF, $nGPU )
  = ("", "",   "",    "",  "",       "",     "",
    1,         "",                999,      "",           1);
GetOptions("in=s" =>\$in,
           "out=s" =>\$out,
           "bad=s" =>\$bad,
           "ref=s" =>\$ref,
           "outputBest=f" => \$maxRMSD,
           "debug" => \$debug,
           "sampleOtherMin" => \$sampleOtherMin,
           "failOnMultiInput" => \$failOnMultiInput,
           "gMinConfSDF=s" => \$gMinConfSDF,
           "nnpOpt=s" => \$nnpOpt,
           "nnpConf=s" => \$nnpConf,
           "nGPU=i" => \$nGPU,
           "omegaOpt=s" => \$omegaOpt) || die $use;
$#ARGV == -1 || die $use;

if( $nGPU > 0 && ! defined($ENV{'CUDA_VISIBLE_DEVICES'}) && $#ARG > 1 )
{  my(@coms) = split(/ /, "mysub.py -q bronze -limit 0:20 -totalMem 15 -nCPU 3 -jobName int.nnpstrain -nGPU 1 -interactive -- tcsh -c");
   warn("No GPU found, using mysub with 20min limit\n\n");
   my($coml) = "source ~cdduser/$ENV{PYTHON_LEVEL}/bin/strain.nnp.csh;"
              ."$0";
   foreach ( @ARG ) { $coml .= " '$_'" }
   push(@coms, $coml);
   warn( join(":",@coms));
   exec(@coms);
}

$in && $out || die $use;
$tmpBase = "/tmp/mmca.$$";
if($ENV{'TMPDIR'}) { $tmpBase = "$ENV{'TMPDIR'}/mmca.$$" }

$ref || ($ref = "$tmpBase.sdf");
$maxRMSD = $maxRMSD * 1;
$omegaOpt = "-canonOrder false $omegaOpt";
$nnpOpt = "-conf $nnpConf $nnpOpt";
$debug && ($nnpOpt = "-logINI debug $nnpOpt");

# @constrVals1 = (".8", ".2", ".05", ".0125");
# distributes median of RMSD at: 0.1, 0.15, 0.2, 0.24, local min is 0.25
#@constrVals1 = ("4", "1", ".25", ".0625");
@constrVals1 = ("1", ".2", ".04", ".008");
@constrVals1 = (".8", ".2", ".05", ".0125");
#@constrVals2 = ("0", "0", "0", "0");
@constrLbls  = ("50", "10", "2", "0.4");

my($constrValsList) = join(",",@constrVals1);

our($sToKcal) = 0.000238845 * 300; # 300 K
*OUT = *STDOUT;
if( $out !~ /^.sdf$/i )
{  open(OUT, ">$out") || die $!;
}

my( $badOut ) = OUT;
if( $bad )
{   open(BAD, ">$bad") || die $!;
    $badOut = BAD;
}

# workaround to OE multiprocessing bug:
delete $ENV{'SLURM_JOBID'};

our( $omegaNNP ) = "$NNPCommand -prune_high_energy 50 .10 $nnpOpt -in .sdf -out .sdf";

open(IN, "sdfTagTool.csh -in $in -out .sdf -remove 'type|deltaE|inRMSD|slack'|") || die "$in $!";

my($count) = 0;

while($_ = <IN>)
{  $rec .= $_;
   if( /\$\$\$\$/ )
   {  if( $failOnMultiInput && $count )
      {  die "-failOnMultiInput was given and more than one struct entered\n";
      }
      open(TMP, ">$tmpBase.sdf") || die $!;
      print TMP $rec;
      close(TMP);

      my( $ok ) = &mmConfAnalysis($tmpBase, $ref, $maxRMSD, $gMinConfSDF);

      my( $myOut ) = OUT;
      if( ! $ok )
      {  $myOut = $badOut;
      }

      open( TMP, "$tmpBase.o.sdf" ) || die $!;
      while( $_ = <TMP> )
      {  print $myOut $_;;
      }
      close(TMP);

      $count++;
      $rec = "";
   }
}
close(IN);


if( ! $debug )
{  unlink( "$tmpBase.no.sdf", "$tmpBase.hc.1.sdf", "$tmpBase.hc.2.sdf", "$tmpBase.constr.sdf",
           "$tmpBase.hc.3.sdf", "$tmpBase.hc.4.sdf", "$tmpBase.hc.5.sdf",
           "$tmpBase.lo.sdf", "$tmpBase.gmin.sdf", "$tmpBase.sdf", "$tmpBase.other.sdf",
           "$tmpBase.o.sdf", "$tmpBase.hopt.sdf", "$tmpBase.omega.sdf", "$tmpBase.sdf",
           "omega2.log", "omega2.parm", "omega2.rpt", "omega2_status.txt",
           "omega2.pending.ism",
           "nnp.log" );
}


if( $bad && -z $bad ) { unlink( $bad ); }

sub mmConfAnalysis
{  my($tmpBase, $ref, $maxRMSD, $gMinConfSDF) = @_;
   my($isOK) = 1;
   my($omegaPID) = 0;
   my( $keepConfsCMD ) = "";
   if( $gMinConfSDF )
   {  $keepConfsCMD = "tee $gMinConfSDF |";
      $keepConfsCMD =~ s/.sdf$/.in.sdf/;
   }

   # find global minimum
   $com = <<COMS;
     #keepConfs# \\
     $omegaNNP \\
     | sdfAlign.pl -in .sdf -out .sdf -ref $ref -rmsdTag inRMSD -method sss -mirror
COMS
   chomp($com);
   $com =~ s/#keepConfs#/$keepConfsCMD/;

   # find global minimum:
   # omega
   # minimize
   # TODO: secondary sort by RMSD to break ties in energy
   $com = <<COMS;
   sdfSmartsGrep.csh -in $tmpBase.sdf -out .oeb -makeHExplicit \\
     | szybki -silent -grad_conv 1 -in .oeb -out .oeb.gz \\
     | omega2 $omegaOpt -in .oeb.gz -out .sdf \\
     | $com \\
     | sdfSorter.csh -in .sdf -out .sdf -numeric -sortTag NNP_Energy_kcal_mol \\
     | tee $tmpBase.omega.sdf \\
     | sdfSplicer.csh -in .sdf -out .sdf -count 1 -readAll \\
     > $tmpBase.gmin.sdf
    set stat=\$status
    if( "$gMinConfSDF" != "" )cp "$tmpBase.omega.sdf" "$gMinConfSDF"
    exit \$stat
COMS


   $debug && warn "\n$com\n";
   unless($omegaPID = fork())   # run in background
   {  exec("tcsh", "-c", $com);
   }

   $com = <<COMS;
   sdfNNP.py -conf $nnpConf -in $tmpBase.sdf -out .sdf \\
     |sdfTagTool.csh -in .sdf -out .sdf -rename 'NNP::Energy_kcal_mol=NNP_Energy_kcal_mol' \\
     |sdfAlign.pl -method sss -in .sdf -ref $ref -out .sdf -rmsdTag inRMSD >$tmpBase.no.sdf &

   $NNPCommand $nnpOpt -in $tmpBase.sdf -out .sdf -constraint heavyAtom \\
     |sdfConformerSampler.csh -in .sdf -out .sdf -maxConfs 20 \\
     |$NNPCommand $nnpOpt -in .sdf -out .sdf -constraint heavyAtom \\
     |sdfSorter.csh -in .sdf -out .sdf -numeric -sortTag NNP_Energy_kcal_mol \\
     |sdfSplicer.csh -in .sdf -out .sdf -count 1 \\
     |sdfAlign.pl -method sss -in .sdf -ref $ref -out .sdf -rmsdTag inRMSD \\
     | tee $tmpBase.hopt.sdf \\
   |$NNPCommand $nnpOpt -in .sdf -out .sdf -harm_constr "0,$constrValsList"\\
     |sdfAlign.pl -method sss -in .sdf -ref $ref -out $tmpBase.constr.sdf -rmsdTag inRMSD

   wait
COMS
   $debug && warn "\n$com\n";
   system($com) == 0 || die "Error in $NNPCommand\n";


   #### split file into single files by contraint
   my(@names) = ("$tmpBase.lo.sdf",  "$tmpBase.hc.1.sdf", "$tmpBase.hc.2.sdf",
                 "$tmpBase.hc.3.sdf","$tmpBase.hc.4.sdf");
   my($n)=0;
   open(INC, "$tmpBase.constr.sdf") || die "$!";
   open(OUT1, ">$names[$n]") || die "$!";
   while($_=<INC>)
   {  print(OUT1 $_);
      if(/\$\$\$\$/)
      {  close(OUT1);
         if( ++$n <= 4) {open(OUT1, ">$names[$n]") || die "$!";}
      }
   }
   close(INC);

   # wait for global min search
   waitpid($omegaPID,0);
   if( $? != 0 ) { die "Error in executing omega subprocess"; }

   ($eGMin,$rmsdGMin) =
       split(/\t/,`sdf2Tab.csh -suppressHead -in $tmpBase.gmin.sdf -tags 'NNP_Energy_kcal_mol|inRMSD'`);
   ($eLMin,$rmsdLMin) =
       split(/\t/,`sdf2Tab.csh -suppressHead -in $tmpBase.lo.sdf -tags 'NNP_Energy_kcal_mol|inRMSD'`);
   $rmsdGMin =~ s/\s//g;
   $rmsdLMin =~ s/\s//g;

   # if local minimum lower in energy than "global" minimum.
   # replace global min with local unless -debug or -bad
   if( (!$debug || !$bad )
      && (! -e "$tmpBase.gmin.sdf" || -z "$tmpBase.gmin.sdf" || $eLMin + .5 < $eGMin) )
   {  warn("\nProblem with golbal minimum: eLMin=$eLMin, eGMin=$eGMin\n\n");
      copy("$tmpBase.lo.sdf", "$tmpBase.gmin.sdf");
      $eGMin = $eLMin;
      $rmsdGMin = $rmsdLMin;

      $isOK = 0;
   }

   # get set of minima which are better than any of the constraint minima
   # either by energy or by rmsd
   my( $grvyCom ) = <<COMS;
      import groovy.transform.Field;
      \@Field double minRMSD = $rmsdGMin;
      \@Field double minE = $eGMin;

       BigDecimal rmsd = f(\$inRMSD);
       BigDecimal e    = f(\$NNP_Energy_kcal_mol);
       if( minRMSD - rmsd >= 0.05 && e - minE + minRMSD - rmsd >= 0.2)
       /*if( rmsd < minRMSD && e - minE + Math.abs(rmsd - minRMSD) > 0.2) */
       {  minRMSD = rmsd;
          minE = e;
          return true;
       };
       return false;
COMS
   $grvyCom =~ s/\n//g;

   $com = <<COMS;
   cat $tmpBase.no.sdf $tmpBase.hopt.sdf \\
         $tmpBase.hc.1.sdf $tmpBase.hc.2.sdf \\
         $tmpBase.hc.3.sdf $tmpBase.hc.4.sdf \\
         $tmpBase.lo.sdf $tmpBase.gmin.sdf \\
   > $tmpBase.o.sdf
   # also pass in constraint minimization so that other minima are also
   # either lower in energy or lower in rmsd than any other result
   cat $tmpBase.o.sdf $tmpBase.omega.sdf \\
     | sdfSorter.csh -in .sdf -out .sdf -numeric -sortTag NNP_Energy_kcal_mol -sortTag inRMSD\\
     | sdfGroovy.csh -in .sdf -out .sdf -c '$grvyCom' \\
     | sdfRMSDSphereExclusion.csh -in .sdf -out .sdf -radius .3 -refFile $tmpBase.o.sdf \\
           -dataSphereFieldName NNP_Energy_kcal_mol -dataSphereRadius 0.5 \\
     | sdfSorter.csh -in .sdf -out .sdf -numeric -sortTag inRMSD \\
     | sdfTagTool.csh -in .sdf -out .sdf -counterTag cntr$$ -addCounter \\
                      -format "type=oth Min{cntr$$}" \\
     | sdfTagTool.csh -in .sdf -out .sdf -remove cntr$$ \\
     > $tmpBase.other.sdf
COMS
   if( $sampleOtherMin )
   {  if( $debug )
      {  warn "\n$com\n";
      }
      system($com) == 0 || die "Error in SDFOptimizer.py\n";
   }

   &appendDeltaE("$tmpBase.no.sdf",    0,   "input", $eGMin);
   &appendDeltaE("$tmpBase.hopt.sdf", .01,   "H opt", $eGMin);
   &appendDeltaE("$tmpBase.hc.1.sdf", 4/$constrLbls[0], "cstr $constrLbls[0]", $eGMin);
   &appendDeltaE("$tmpBase.hc.2.sdf", 4/$constrLbls[1], "cstr $constrLbls[1]", $eGMin);
   &appendDeltaE("$tmpBase.hc.3.sdf", 4/$constrLbls[2], "cstr $constrLbls[2]", $eGMin);
   &appendDeltaE("$tmpBase.hc.4.sdf", 4/$constrLbls[3], "cstr $constrLbls[3]", $eGMin);
#   &appendDeltaE("$tmpBase.hc.5.sdf", 4/$constrLbls[4], "cstr $constrLbls[4]", $eGMin);
   &appendDeltaE("$tmpBase.lo.sdf",    5, "loc Min", $eGMin, $sGMin);
   &appendDeltaE("$tmpBase.gmin.sdf", 10, "glb Min", $eGMin, $sGMin);

   $sampleOtherMin && &appendDeltaE("$tmpBase.other.sdf", 9, "", $eGMin, $sGMin);

   $com = "";
   $inputConfCom = <<COMS;
      sdfTagTool.csh -in $tmpBase.no.sdf -out .sdf \\
         -remove "AlignSize|AlignPct" \\
         -reorder 'type|inRMSD|deltaE' \\
      |sdfGroovy.csh -in .sdf -out $tmpBase.o.sdf -c \\
        '\$>TITLE="| RMSD|   dE   |  input       |";' ;
COMS
   if( $maxRMSD < 999 )
   { # output only lowest e pose with rmsd<maxRMSD
     $com = "|sdfSorter.csh -in .sdf -out .sdf -numeric -sortTag deltaE"
           ."|sdfGroovy.csh -in .sdf -out .sdf -c 'return (f(\$inRMSD)<=$maxRMSD)'"
           ."|sdfSplicer.csh -in .sdf -out .sdf -count 1";
     $inputConfCom = "rm $tmpBase.o.sdf;";
   }
   $com = <<COMS;
      $inputConfCom
      cat $tmpBase.hopt.sdf \\
         $tmpBase.hc.1.sdf $tmpBase.hc.2.sdf \\
         $tmpBase.hc.3.sdf $tmpBase.hc.4.sdf \\
         $tmpBase.lo.sdf $tmpBase.gmin.sdf $tmpBase.other.sdf \\
     |sdfTagTool.csh -in .sdf -out .sdf \\
         -remove "AlignSize|AlignPct" \\
         -reorder 'type|inRMSD|deltaE' \\
     |sdfGroovy.csh -in .sdf -out .sdf -c \\
       '\$>TITLE=String.format("| %5s | %5s | %-10s |", \$inRMSD, \$deltaE, \$type)' \\
     $com >>$tmpBase.o.sdf
COMS
   $debug && warn "\n$com\n";
   system($com) == 0 || die "Error in summarizing\n";

   return $isOK;
}



sub appendDeltaE
{  my( $fName, $slack, $type, $eMin) = @_;
   my( $sTxt, $eT ) = ("", 0);

   open( TMPIN, $fName ) || die "$! ($fName)";

   while( $_ = <TMPIN> )
   {  if( /\$\$\$\$/ )
      {  $sTxt .= getDelatE( $eT, $slack, $type, $eMin);
         $sTxt .= "\$\$\$\$\n";
         $eT = 0;
         next;
      }

      $sTxt .= $_;

      if( /<NNP_Energy_kcal_mol>/ )
      {  $_ = <TMPIN>;
         $sTxt .= $_;
         chomp;
         $eT = $_;
      }
   }
   close(TMPIN);

   open(TMPIN, ">$fName" ) || die $!;
   print TMPIN $sTxt;
   close(TMPIN);
}



sub getDelatE
{  my( $eT, $slack, $type, $eMin) = @_;
   my( $fTxt ) = "";

   $type && ( $fTxt  = sprintf("> <type>\n%s\n\n",$type));
   $fTxt .= sprintf("> <deltaE>\n%0.2f\n\n", ($eT-$eMin));
   $fTxt .= sprintf("> <slack>\n%s\n\n", $slack);

   return $fTxt;
}
