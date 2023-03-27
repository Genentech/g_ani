#!/usr/bin/env perl
use warnings;

$test = <<'TEST';

cat ~/dev/ml_qm/data/4667.sdf ~/dev/ml_qm_data/4667.sdf \
| scripts/sdfNNPConfSample.pl -in .sdf -out t.sdf \
   -nnpOpt "-maxiter 10 -nGPU 1 -trust 1"

or

mysub.py -limit 0:20 -totalMem 20 -nCPU 2 -nGPU 1 -jobName testNNPConfA <<'coms'
#!/bin/tcsh -f
   source /system/gredit/clientos/etc/profile.d/lmod.csh;
   source ~cdduser/prd/bin/strain.nnp.csh
   time sdfNNPConfAnalysis.pl -in ~/dev/ml_qm_data/4667.sdf -out 4667.nnpconfa.sdf -debug
'coms'
TEST
$script=$0;
$script =~ /^\// || ( $script="$ENV{PWD}/$script" );

use File::Copy;
use Getopt::Long;
use POSIX ":sys_wait_h";

#changes due to experiments with macrocycle
my($omegaOpt ) = "-includeInput true -sampleHydrogens -searchff mmff94s -ewindow 50 -maxconfs 650 ";
my($NNPCommand) = "sdfMOptimizer.py";
my($nnpOpt) = "-maxiter 2000 -nGPU 1 -trust 1";
my($nnpConf) = "~smdi/prd/ml_qm_data/nnp/ani22/bb/bb.3.json";

$use = <<USE;
sdfNNPConfSample.pl [-ref alignRef.sdf] [-nnpOpt s] [-omegaOpt s] [-nnpConf s]
                     -in in.sdf -out out.sdf
  -ref  .............. align conformers to given 3d sss fragment
  -nnpConf ........... NNP .json file or directory for ANI
  -nnpOpt ............ additional option for the NNP. default "$nnpOpt"
                       local optimization will allways use -conj
  -omegaOpt .......... additional option for omega. def "$omegaOpt"
  -generate3D ........ generate initial 3D conf
  -debug ............. print commands before executing

If used with sdfMultiplexer use the -groupByAtomCount to make sure the records
for one input structre are kept together.
USE

if( ! defined($ENV{'CUDA_VISIBLE_DEVICES'}) && $#ARGV > 1 && ! grep { /-h/i } @ARGV )
{  my(@coms) = split(/ /, "mysub.py -limit 0:20 -totalMem 20 -nCPU 2 -jobName int.nnpstrain -nGPU 1 -interactive -- tcsh -c");
   warn("No GPU found, using mysub with 20min limit\n\n");
   my($coml) = "source ~cdduser/$ENV{PYTHON_LEVEL}/bin/strain.nnp.csh;"
              ."$0";
   foreach ( @ARGV ) { $coml .= " '$_'" }
   push(@coms, $coml);
   warn( join(":",@coms));
   exec(@coms);
}

my( $com, $in, $ref, $out, $generate3D, $debug,)
  = ("", "",   "",    "",  "",          "" );
GetOptions("in=s" =>\$in,
           "out=s" =>\$out,
           "ref=s" =>\$ref,
           "debug" => \$debug,
           "generate3D" => \$generate3D,
           "nnpOpt=s" => \$nnpOpt,
           "nnpConf=s" => \$nnpConf,
           "omegaOpt=s" => \$omegaOpt) || die $use;
$#ARGV == -1 || die $use;

$in && $out || die $use;
$tmpBase = "/tmp/mmca.$$";
if($ENV{'TMPDIR'}) { $tmpBase = "$ENV{'TMPDIR'}/mmca.$$" }

$omegaOpt = "-canonOrder false $omegaOpt";
$nnpOpt = "-conf $nnpConf $nnpOpt";
$debug && ($nnpOpt = "-logINI debug $nnpOpt");

if( $out !~ /^\.[^.]+(\.gz)?$/ )
{  open(STDOUT,'>',$out) || die "can't open $out: $!";
   ($outType) = $out =~ /(.[^.]+(\.gz)?)$/;
} else
{  $outType = $out;
}

# workaround to OE multiprocessing bug:
delete $ENV{'SLURM_JOBID'};

our( $omegaNNP ) = "$NNPCommand -prune_high_energy 40 .15 $nnpOpt -in .sdf -out .sdf";

open(IN, "sdfTagTool.csh -in $in -out .sdf -remove 'deltaE'|") || die "$in $!";

my($count, $alignPID ) = (0, 0);
# alignPID is the process ID of the alignment pip for the previous molecule
# it runs in the background but we must wait for completion before outputing
# next mol

while($_ = <IN>)
{  $rec .= $_;
   if( /\$\$\$\$/ )
   {
      $alignPID = &mmConfSample($outType, $rec, $ref, $generate3D, $tmpBase, $count, $alignPID);

      $count++;
      $rec = "";
   }
}
close(IN);
&waitForAlign($alignPID, $tmpBase, $count-1);


if( ! $debug )
{  unlink( "omega2.log", "omega2.parm", "omega2.rpt", "omega2_status.txt",
           "szybki.param", "szybki.log", "szybki.status",
           "oeomega_classic.parm", "oeomega_classic.rpt",
           "oeomega_classic_status.txt", "oeomega_classic.log",
           "omega2.pending.ism", "nnp.log" );
}


sub mmConfSample
{  my($outType, $rec, $ref, $generate3D, $tmpBase, $count, $alignPID) = @_;
   my($omegaPID) = 0;

   # find global minimum
   my($com) = $omegaNNP;
   $generate3D && ($generate3D = "|omega2 -in .sdf -out .sdf -strictstereo false -maxconfs 1");

   # find global minimum:
   # omega
   # minimize
   $com = <<COMS;
   cat <<'SDFNNPCA123'\\
     $generate3D |sdfSmartsGrep.csh -in .sdf -out .oeb -makeHExplicit \\
     | szybki -silent -grad_conv 1 -in .oeb -out .oeb.gz \\
     | omega2 $omegaOpt -in .oeb.gz -out .sdf \\
     | $com \\
     > $tmpBase.$count.sdf
$rec
'SDFNNPCA123'
COMS

   $debug && warn "\n$com\n";
   system("tcsh", "-c", $com) == 0 || die "Error in $com: \n$?\n";

   # wait for alignment pipe form last recored to complete before starting new one
   &waitForAlign($alignPID, $tmpBase, $count-1);

   $com = "";
   if( $ref )
   {   $com = "| sdfAlign.pl -in .oeb -out .oeb -ref $ref -rmsdTag inRMSD -method sss";
   }

   $com = <<COMS;
     cat $tmpBase.$count.sdf \\
     | sdfSorter.csh -in .sdf -out .oeb -numeric -sortTag NNP_Energy_kcal_mol \\
     | sdfRMSDSphereExclusion.csh -in .oeb -out .oeb -radius 0.3 -useMaxDeviation \\
           -dataSphereFieldName NNP_Energy_kcal_mol -dataSphereRadius 0.15 \\
     | sdfAggregator.csh -in .oeb -out .oeb -outputmode all \\
          -function 'nnpminE=min(NNP_Energy_kcal_mol)' \\
     | sdfGroovy.csh -in .oeb -out .oeb -c '\$>deltaE=f(\$NNP_Energy_kcal_mol)-f(\$nnpminE)' \\
     $com \\
     | sdfTagTool.csh -in .oeb -out $outType -remove 'nnpminE|MMFF VdW|MMFF Coulomb|MMFF Bond|MMFF Bend|MMFF StretchBend|MMFF Torsion|MMFF Improper Torsion|Ligand MMFF Intramol. Energy|Total_energy|sphereIdx|includeIdx|centroidRMSD'
COMS

   $debug && warn "\n$com\n";
   unless($alignPID = fork())   # run in background
   {  exec("tcsh", "-c", $com);
   }
   return $alignPID;
}

sub waitForAlign
{  my($alignPID, $tmpBase, $count) = @_;

   if( ! $alignPID ) { return; }

   waitpid($alignPID,0);
   {  if( $? != 0 ) { die "Error in previous alignment pipe"; }
      $debug || unlink( "$tmpBase.$count.sdf" );
   }
}
