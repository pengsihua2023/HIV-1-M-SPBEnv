# HIV-SPBEnv
## Requirements
Python == 3.9.18    
torch == 2.1.0+cu121  
torchvision ==  0.16.0+cu121  
scikit-learn == 1.3.2  
pandas == 2.1.2  
numpy == 1.24.1  
matplotlib == 3.8.0  
biopython == 1.79  

### Dependencies can be installed using the following command:
conda create -n HIV-SPBEnv python=3.9.18  
conda activate HIV-SPBEnv  

pip install -r requirements.txt  

## Usage For AI Researchers
You can run it from the command line  

    cd ./HIV-SPBEnv  
    python extract_feature.py  

## Usage For Biomedical Researchers
We deployed a trained model on a dedicated server, which is publicly available at http://www.peng-lab.org:5000/analysis, to make it easy for biomedical researcher to perform HIV 1 subtype classification in their research.

Users can upload their env gene sequences of HIV to the server, and then they can quickly obtain the predicted results of the HIV 1 subtype classification.

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

