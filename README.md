# HIV-SPBEnv
## Requirements
Python == 3.6.13  
opencv-python == 4.5.1.48  
tensorflow-gpu == 1.14.0  
scikit-learn == 0.24.0  
pandas == 1.1.5  
numpy == 1.19.5  
matplotlib == 3.3.4  
h5py == 2.10.0  
Bio == 1.3.9  

### Dependencies can be installed using the following command:
conda create -n DeepmRNALoc python=3.6.13  
conda activate DeepmRNALoc  

pip install -r requirements.txt  
HIV-SPBEnv is a project for HIV 1 subtype classification.
The detailed composition of the DNA sequence data of HIV env gene is shown in Table 1.




### Table 1 The original data set of the 12 subtypes of HIV env DNA sequences.
| Subtype |Sample Size | Subtype |Sample Size |
|-------|---------|-------|---------|
| A1 | 311 | F2 |16 |
| A2 | 5 | G |136 |
| B | 2,887 | H |10 |
| C | 1,717 | J |5 |
| D | 145 | K |2|
| F1 | 73 | L |3 |
### Table 2 The training data set for model training after augmentation.
| Subtype |Sample Size | Subtype |Sample Size |
|-------|---------|-------|---------|
| A1 | 1,000 | F2 |1,000 |
| A2 | 1,000 | G |1,000 |
| B | 1,000 | H |1,000 |
| C | 1,000 | J |1,000 |
| D | 1,000 | K |1,000|
| F1 | 1,000 | L |1,000 |
### Table 3 The Independent testing data set for model evaluation.
| Subtype |Sample Size | Subtype |Sample Size |
|-------|---------|-------|---------|
| A1 | 300 | F2 |300 |
| A2 | 300 | G |300 |
| B | 300 | H |300 |
| C | 300 | J |300 |
| D | 300 | K |300|
| F1 | 300 | L |300 |

