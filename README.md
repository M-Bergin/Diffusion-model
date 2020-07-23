# Diffusion-model

MATLAB code to perform a Bayesian approach to fitting non-fickian diffusion data.

The main 1D diffusion simulation is performed in the function Diffusion_numeric_1D_mid.

The other two higher level scripts call the diffusion function to compare the model to real data that is fed in through the text files. To repeat the analysis, the path to the data needs to be edited to match the file structure you choose. The scripts consist of multiple for loops over each free parameter and therefore while the individual diffusion calculations are fast, the entire script takes a long time to execute.

The scripts were most recently used on a HPC to produce publication quality graphs, with each script taking over 100 CPU hours to complete. Therefore, the number of points to be used will need to be tweaked for other uses.
