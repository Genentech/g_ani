# g_ANI

 g_ANI contains a reimplementation of the ANI Neural Net Potential developed at Genentech.
It also contains a command line tool [sdfNNPConfAnalysis.pl](iscripts/sdfNNPConfAnalysis.pl)
computes the strain in a small molecule ligand confomation. This allows the computation of strain energy with QM accuracy in just a few minutes on a GPU while analyzing hundreths of conformations.

Relevant references are:
   - [ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost](https://pubs.rsc.org/en/content/articlelanding/2017/SC/C6SC05720A)
   - [chemalot and chemalot_knime: Command line programs as workflow tools for drug discovery](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0228-9)
   - [Conformational energy penalties of protein-bound ligands.](https://link.springer.com/article/10.1023/A:1008007507641)

## Installing
   - Install git large file support:
     ```
     git lfs install
     ```
   - clone these three packages into one directory
     ```
        git clone https://github.com/Genentech/cdd_chem.git
        git clone https://github.com/Genentech/t_opt.git
        git clone https://github.com/Genentech/g_ani.git
      ```
  - install the tools from the [Autocorrelator](https://github.com/chemalot/autocorrelator) package and make sure they are avaialble on your path.
  - install the tools from the [Chemalot](https://github.com/chemalot/chemalot) package and make sure they are avaialble on your path.
  - create the conda package
   
     you might need to issue `conda config --set channel_priority strict`
     ```
     cd g_ani
     conda env create --file requirements_dev.yaml -n g_ANI
     ```
   - activate the conda package:
      ```
      conda activate g_ANI
      ```
   - Install the auxilliare packages indevelopment mode as long as you are developing.:
      ```
      cd ../cdd_chem/
      pip install -e .
      cd ../t_opt/
      pip install -e .
      ```
   - To run the strain energy calculation you need the following tools from [OpenEye](https://www.eyesopen.com/).<br/>
     Note: you can run minimizations using sdfMOptimizer.py and singlepoint calculations with sdfNNP.py without openeye tools
           or licenses as the cdd_chem package will use RDKit if OEChem is not available.
   
       - oemga
       - szybki

   - All NNP calculations will run significanlty faster if you have a GPU available on your computer. If a GPU is not avaialble the slowe CPU implementation will be used.

   - You will need a ~8GB of memory to run the esample below. Be aware that to little memory will cause errors with unexpected messages.
   - Run test example:
   ```
   scripts/sdfNNPConfAnalysis.pl -in tests/data/CCCC.sdf -out out.sdf -sampleOtherMin
   ```
   The input file contains two conformations of butane. The output file will contain 16 conformations. 8 for each input conformation evaluating the strain and highlighting the areas of the input conformation with the highest strain (cf. below).

## Strain Energy Calculation
### Method
This implementation follows the definition by Boström et.al. (Conformational energy penalties of protein-bound ligands, JCAMD 1998).

- A conformational search is performed by enumerating conformers with oemga and mimizing them with the NNP. By default up to 650 conformations are enumerated to be complete.
- The conformation with the lowest energy is identified and defines the **"global minimum"** conformation. All energies reported are relative to this conformation.
- The input conformation is minimized to yield the **"local minimum"**
- The hydrongens are minimized on the input conforamtion to yield the **"Hydrogen optimized""** conformation (HOpt).
- The hydrogen optimized conformation is optimized using 4 decreasing quadratic retaining potentials. This will yeild coformations with increasing relaxation and decreasing relative energy. The relaxiation will happen along the most contraiend part of the molecule, thus highlighting areas of strain. These are the **"constrained minima"**
- If the `-sampleOtherMin` option is given **"Other minima"** are returned. these minima are on a pareteo minmum curve that optimizes RMSD or relative energy. These allow the identification of minima that are similar to the input conformation and have low relative energy or low RMSD.

By looking at the conformations from first to last the relaxation along the most strained features of the input conformations can be identified. Order is as follows:

1. Input Conformation
2. Hydrogen atoms optimized input (Hopt)
3. 4 constrained minimized conformation with decreasing constrains
4. Local minimum
5. Global minimum<br/>
   The global minimum will frequently be ill aligned to the input conformation as it will generally have a very different conformation.
6. Other minima (as described above)<br/>
   These minima should be evaluated to see if there are minor changes that preserve the overal structure but largely reduce the strain.
   
The output file will contain the following fields:

   - **type** one of the following specifying the type of this conformer (cf. above):<br/>
      input, H Opt, cstr 50, cstr 10, cstr 2, cstr 0.4, loc Min, glb Min, oth *
   - **inRMSD** rmsd of this conformation to the input conformation [A]
   - **deltaE** relative energy of this conformation ot the global minimum [kcal/mol]
   - **NNP_Energy_kcal_mol** absolute energy of this conformation as computed with the g_ani NNP.

### Example

The following ligand conformation was retrieved from the PDB ([5BVF](https://www.rcsb.org/structure/5BVF)) of a small molecule bound to ERK2. This is one of the few compound that is bound in what is deamed a high energy conformation.

![5BVF Pocket](documentation/5bvf/5bvf_pocket.jpg)
<br/>Ligand bound to ERK2 in 5VFB
<table><tr><td align="center"><img src='documentation/5bvf/5bvf.gif'/></td>
           <td align="center"><img src='documentation/5bvf/5bvf_WithInput.gif'/></td></tr>
       <tr><td align="center">Strain analysis animation</td>
           <tdalign="center">Strain analysis animation with overlay of input</td></tr>
       <tr><td colspan='2'>
         Result of strain energy computaiton on ligend in 5BVF. As can be seen the largest train is on the phynly ring on the right. It clearly relaxes from the position in the input conformation to the conformation in the contraint minima (cnst 0.4). Note that the global minimum exhebits an intramolecular hydrogen bond. The energy of the global minimum is probably not reflective of the energy in solution phase as the NNP was traiend on gas phaseDFT calculations. Thus the strain in this calculation might be overestimated.</br>
         RMSD to Input [A]. Relative Energy (dE) to Global Minimum [kcal/mol]</td></tr></table>
           
### Explaination
The constraint minimization account for multiple non-phisical sources of strain:

- Differnces between the method used to genrate the input confomation and the NNP used in evaluatingthe striain. Small changes int the bond length deemed to be optimal between two atoms would yield very hi energie differences. Allowing the slight relaxation will remove this artificat.
- Molecular flexibility of the protein and ligand always allow for some movement.
- Crytal structure refinement has an intrinsic uncertainty.

### Statistics

We have computed the strain energy with the this method for 750 neutral ligands from PDB database with good resolution.
The follwoing boxplot shows the distribution of the strain energy of these 750 conformations for different values of maximum relaxation. E.g. the box at 0.4 A maxRMSD was coputed by applying the method described above. For each of the 750  input conformation. Only conformations within 0.4 A were retained and the lowest relative energy is reported.

![BoxPlot](documentation/5bvf/BoxPlot.jpg)

As can be expected the more relaxation is allowed the lower the strain energy is. In looking at many strain energy calculation we have determined that a relaxaion of 0.4 A results in a conformation that is very close to the input conformation but in which many artifacts causeing strain have been released. We therefore recomend looking at hte neergy of the confrmations with less than 0.4 A deviation form the input first. If the lowest energy of these conformation is below 2-3 kcal/mol the conformation is considered to have a low strain energy. For conformations with strain energys at 0.4A > 3 kcal/mol we recomend looking at the relaxation pattern and  trying to undestand which parts of the molecule are contributing most to the strain. Structural changes to the molecule should be considered to reduce the streain. The statistics above suggest that compounds with strains (at 0.4A) > 2-3kcal/mol have a small likeihood of being consistent with chrystallographically observed conformations.

### comparison to ForceField based implementation
We ran the same strain anergy computation using the MMFF94S forcefield using the sheffiled solvation model instead of the NNP on the 750 confomation form the PDB described above. The follwoing graph comparese the results:

[gANI vs MMFF94S](documentation/GANNI_MMFF.jpg)

As expected both method classify most confromations from the pdb as low strained. However, for some conformations differences highlight limitations of either method.
<table>
 <tr><th colspan='4'>Conformations strained according to MMFF94 but not strained accoring to gANI</th></tr>
 <tr>
  <td>[GANNI strained](documentation/MMFF_strained.jpg)</td>
  <td>>[5jn8](documentation/5jn8.jpg)</td>
  <td>>[4dvi](documentation/4dvi.jpg)</td>
  <td>>[5tz3](documentation/5tz3.jpg)</td>
 </tr>
<tr>
 <td>[Ledgend](documentation/confColors.jpg)</tr>
  <td>Inthe crystallographic pose of 5jn8 the carbonly oxygen is pointing towards the thiadiazole sulfur. This conformation is stabilized by the favorable O-S interaction. This is reproduced by gANI but not by MMFF94S. It is well known that O-S interactions are frequently seen as repulsive by force fields.</td>
  <td>In casee of the 4dvi ligand both the gANI and the MMFF94 conformations differ from the crystallographic conformation on the central phenyl ring. The energy difference for the gANI conformation is computed to be just 0.1 kca/mol while the MMFF94 Force Feld predicts a difference of 6 kcal/mol.</td>
  <td>For 5tz3 both the gANI and the crystalographic conformations are mostly planar with a hydrogen bond between the amide NH and the 5 memberd ring nitrogen. In the MMFF94S conformation this interaction is not made and the conformation is twisted out of plane. Oru assumption is that the hydrogen bonding conformation is to strained in the MMFF94S computation due to teh close distance required by the ridgide backbone of the compound.</td>
 </tr>
 
 
 <tr><td colspan=4'4'/></tr>
 <tr><th colspan='4'>Conformations strained according to gANI but not strained accoring to MMFF94</th></tr>
 <tr>
  <td>[GANNI strained](documentation/GANNI_strained.jpg)</td>
  <td>>[2ori](documentation/2ori.jpg)</td>
  <td>>[5lrd](documentation/5lrd.jpg)</td>
  <td>>[5xs2](documentation/5xs2.jpg)</td>
 </tr>
 <td>[Ledgend](documentation/confColors.jpg)</tr>
  <td colspan='2>Fro 2ori and 5lrd the gANI minimum conformation deviates significanly from the crysallographic conformation and makes an intramolecular hydrogen bond. The strength of this hydrogen bond is overestimated by gANI as the NNP was rained on gasphase DFT energies.</td>
  <td>The gANI and MMFF94S conforamtion of 5xs2 differ in the orientation of the amide group. Both conformations are difficult to differntiate based on the elecron density. The conformation predicted by gANI however places the carbonyl oxigen next to the electropositive hydrogen no the pyrole N.</td>
 </tr>
</table>

## Features

-  Contains multiple model architechtures

## Credits

This Python package was created with

- [Cookiecutter](https://github.com/audreyr/cookiecutter)
- [CDD PyPackage](https://code.roche.com/SMDD/python/cdd-pypackage) project template
