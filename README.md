# g_ANI

 g_ANI contains a reimplementation of the ANI Neural Net Potential developed at Genentech.
It also contains a command line tool [sdfNNPConfAnalysis.pl](iscripts/sdfNNPConfAnalysis.pl)
computes the strain in a small molecule ligand confomation.

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



## Features

-  Contains multiple model architechtures ant our trials in pKa prediction and Docking Pose assesment

## Credits

This Python package was created with

- [Cookiecutter](https://github.com/audreyr/cookiecutter)
- [CDD PyPackage](https://code.roche.com/SMDD/python/cdd-pypackage) project template
