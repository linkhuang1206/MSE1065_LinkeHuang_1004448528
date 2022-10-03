# MSE1065_LinkeHuang_1004448528
for homework submission and course project version control

# Data selection
## Dataset description
The data is downloaded from Citrination datasets named "Electrocatalysts for CO2 reduction and H2 evolution from Ulissi group", available at:  
https://citrination.com/datasets/193373/show_files  
The origin of the dataset is from the paper "Active learning across intermetallics to guide discovery of electrocatalysts for CO2 reduction and H2 evolution", available at:  
https://doi.org/10.1038/s41929-018-0142-1
More detailed information and associated codes are inluded in the GitHub page of the work, available at:  
https://github.com/ulissigroup/GASpy_manuscript  
The data contains the crystal structure info and the adsorption info of 23,141 adsorption sites from 1,499 intermetallic crystal for hydrogen evolution reaction(HER)

## Featurization
The dataset are to be featurzied by crystal structure(lattice vectors, atomic position), adsorption site atom & associate neigbour list and the elements are to be featurized by atomic number,electronegativity and the # of the atoms of the element coordinated with the adsorbate. Depending on the availability of comupational resources, the feature might be further simplified.
The goal of this project would be to develop a classfication model that can predict whether the material can have a near_optimal adsorption energy(-0.37eV to -0.17eV), and the result would be validated by comparing with the identified near-optimal surfaces. 
