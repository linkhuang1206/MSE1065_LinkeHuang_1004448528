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
The data contains the crystal structure info and the adsorption info of 22,675 adsorption sites from 995 intermetallic crystal for hydrogen evolution reaction(HER)

## Featurization
The dataset are to be featurzied by 1. bulk compositional information based on the alloy module, 2. adsorption site atom & associate neigbour list and the elements(encoded by atomic number, electronegativity and the coordination number). 

The goal of this project would be to develop a regression model that can predict the adsorption energy of the material(-0.37eV to -0.17eV) from the feature.
