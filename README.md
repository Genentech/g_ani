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

     git lfs install

   - clone these three packages into one directory

     git clone https://github.com/Genentech/cdd_chem.git

     git clone https://github.com/Genentech/t_opt.git

     git clone https://github.com/Genentech/g_ani.git

   - create the conda package
     you might need to issue `conda config --set channel_priority strict`

     cd g_ani

     conda env create --file requirements_dev.yaml -n g_ANI


## Features

-  Contains multiple model architechtures ant our trials in pKa prediction and Docking Pose assesment

## Credits

This Python package was created with

- [Cookiecutter](https://github.com/audreyr/cookiecutter)
- [CDD PyPackage](https://code.roche.com/SMDD/python/cdd-pypackage) project template
