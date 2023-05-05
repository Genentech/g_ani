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

## Strain Energy Method
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
![5BVF Pocket](../documentation/5bvf/5bvf_pocket.jpg)

## Features

-  Contains multiple model architechtures

## Credits

This Python package was created with

- [Cookiecutter](https://github.com/audreyr/cookiecutter)
- [CDD PyPackage](https://code.roche.com/SMDD/python/cdd-pypackage) project template
