# HIV-SPBEnv  
HIV-SPBEnv is a project for HIV 1 subtype classification.  
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
## Results
HIV-SPBEnv was trained by using the original dataset (Table 1) plus the augmentaion dataset (Table 2).  HIV-SPBEnv's classifcation accuracy was 100% for the independent dataset (Table 3).
## The data sets
### The original dataset
The detailed composition of the DNA sequence data of HIV env gene is shown in Table 1. For this data set, the sample size of some subtypes is too small, and there are only 2-5 samples in 4 subtypes. It is obviously impossible to build a machine learning model on such data. Then the only way is to find ways to increase the sample size for the small sample size subtypes.  
     
### Table 1 The original data set of the 12 subtypes of HIV env DNA sequences.
| Subtype |Sample Size | Subtype |Sample Size |
|-------|---------|-------|---------|
| A1 | 311 | F2 |16 |
| A2 | 5 | G |136 |
| B | 2,887 | H |10 |
| C | 1,717 | J |5 |
| D | 145 | K |2|
| F1 | 73 | L |3 |
### The Training dataset
### Table 2 The training data set for model training after augmentation.  
| Subtype |Sample Size | Subtype |Sample Size |
|-------|---------|-------|---------|
| A1 | 500 (300) | F2 |500 (14) |
| A2 | 500 (5) | G |500 (130) |
| B | 500 (500) | H |500 (8) |
| C | 500 (500) | J |500 (5) |
| D | 500 (120) | K |500 (2)|
| F1 | 500 (70) | L |500 (3) |
### The evaluation dataset
### Table 3 The Independent testing data set for model evaluation.
| Subtype |Sample Size | Subtype |Sample Size |
|-------|---------|-------|---------|
| A1 | 100 | F2 |100 |
| A2 | 100 | G |100 |
| B | 100 | H |100 |
| C | 100 | J |100 |
| D | 100 | K |100|
| F1 | 100 | L |100 |

