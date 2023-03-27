#alberto



NumToName = \
    [ '*',
      'H',                                                             'He',
      'Li', 'Be',                    'B',   'C',   'N',   'O',   'F',   'Ne',
      'Na', 'Mg',                    'Al',  'Si',  'P',   'S',   'Cl',  'Ar',
      'K',  'Ca',  'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                                     'Ga',  'Ge',  'As',  'Se',  'Br',  'Kr',
      'Rb', 'Sr',  'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                                     'In',  'Sn',  'Sb',  'Te',  'I',   'Xe'
      ]


NameToNum = { NumToName[n]: n for n in range(len(NumToName))}

# vdw: Batsanov, 2001, table 1 (Bondi preffered), wikipedia (if missing)
NumToVDWRadius = \
    [ 0.,
      1.2,                                          1.4,
      1.82, 1.53, 1.92, 1.7,    1.5,  1.4,  1.35,  1.54,
      2.27, 1.73, 1.84, 2.1,    1.8,  1.85, 1.75,  1.88,
      ]
