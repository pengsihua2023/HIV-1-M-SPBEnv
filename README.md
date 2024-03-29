# HIV-1-M-SPBEnv  
The Human Immunodeficiency Virus, also known as HIV, is a retrovirus that causes deficiencies in the human immune system. This virus attacks and gradually destroys the human immune system, leaving the host unprotected during infection. People who are infected with HIV and pass away often die from secondary infections or cancer. AIDS is the final stage of HIV infection.

There are two main types of the AIDS virus: HIV-1 and HIV-2. HIV-1 originated in the area around the Congo Basin in Africa and is the most prevalent strain globally. It is responsible for about 95% of all HIV infections. HIV-2 is primarily found in West Africa, although it also affects a small number of people in Europe, India, and the United States.

HIV-1 can be divided into 4 groups: M, N, O and P, and Group M is the most widely distributed worldwide, which is divided into 12 different subtypes or sub-subtypes, namely A1, A2, B, C, D, F1, F2, G, H, J, K, and L. The formation of various subtypes or sub-subtypes of HIV-1 group M is the result of continuous molecular evolution. Correct classification of subtypes or sub-subtypes is important for vaccine design, therapeutic drugs, and effective prevention and control of AIDS in the global community.

Precise M subtype or sub-subtype classification relies on phylogenetic analysis of specific gene sequences. In the past, subtypes were roughly determined through homologous searches in the NCBI database. The accuracy depended on the searcher's judgment level, and sometimes even the correct judgment could not be made. Classification of HIV-1 subtype based on statistical modeling methods has also been developed, but due to the small sample size of some subtypes, this has caused great limitations in the tools derived from statistical modeling. Therefore, I developed a deep learning-based method, which I named HIV-1-M-SPBEnv.  

To address the issue of samples being scarce for some subtypes, I successfully used artificial genetic mutation methods to synthesize new machine learning samples, thereby cleverly solving the problem of insufficient samples for some subtypes.

Due to the rapid evolution rate of the env gene, I chose to model the env gene sequence samples for the classification of HIV-1 subtypes. I download env gene sequences at the HIV Sequence Database supported by Los Alamos National Laboratory ([https://www.hiv.lanl.gov/content/index](https://www.hiv.lanl.gov/content/sequence/HIV/mainpage.html)). I downloaded the 2021 version of the env gene sequence, which is the latest version so far, with a total sample size of 5,310 (Table 1).    
  
In my deep learning model framework, I use the Kmer method to vectorize DNA sequences; The model training adopts a unique strategy. The function of the Autoencoder is to extract high-dimensional feature information from DNA sequences. There are two training cycles, including Autoencoder training cycle and classifier training cycle. In the Autoencoder training cycle, I train the Autoencoder with the criterion of making the output reconstruction loss approach zero, thereby enabling the output signal to completely reconstruct the input signal, while in the classifier training cycle, the goal is to train a classifier to minimize its training loss, thereby achieving higher classification accuracy. To this end, the two loss function values are added together to obtain a total loss function, and this total loss function is made to approach zero after the completion of training; this serves as the training criterion for the entire model. With the total loss approaching zero, the Autoencoder's reconstruction loss also tends towards zero, then achieving a perfect reconstruction of the Autoencoder's output to its input. Consequently, the encoder's output thus perfectly and losslessly extracts high-dimensional feature information of DNA sequences, representing the high-dimensional information of DNA sequences with low-dimensional information. Subsequently, the output signal of the encoder is fed into the full connected neural network block, which is used for the classification task. The architecture of HIV-1-M-SPBEnv is shown in Figure 1.  
Using an independent validation dataset, the accuracy of HIV-1-M-SPBEnv reached 100%, demonstrating strong model generalization capabilities. The trained model is deployed at http://www.peng-lab.org:5000/.    
## 1. Requirements
Python == 3.10.13    
torch == 2.1.0+cu121  
torchvision ==  0.16.0+cu121  
scikit-learn == 1.3.2  
pandas == 2.1.2  
numpy == 1.24.1  
matplotlib == 3.8.0  
biopython == 1.79  

## 2. Create virtual enviroment and install Pytorch
The following commands only use the Windows system as an example.  
````
conda create -n HIV-1-M-SPBEnv python=3.9.18  
conda activate HIV-1-M-SPBEnv  

# My GPU Cuda version is 12.3. However the latest version of Cuda compatible with Pytorch is Cuda 1.21.  
# Please check at https://pytorch.org/ for appropriate installation commands for your computer.  

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# If you do not have a GPU card on your computer, please use the following command to install Pytorch:

pip3 install torch torchvision torchaudio 

````
## 3. Download HIV-1-M-SPBEnv, and install dependencies and HIV-1-M-SPBEnv.
#Clone the HIV-1-M-SPBEnv repo
````python
git clone https://github.com/pengsihua2023/HIV-1-M-SPBEnv.git
cd HIV-1-M-SPBEnv

# Install requirements
pip install -r requirements.txt  

#Install HIV-1-M-SPBEnv
pip install .
````
## 4. Run the HIV-1-M-SPBEnv
````python
#Just execute the command "hiv-env" in any command line environment. 
hiv-env
````  
## 5. Results
After running the "hiv-env" command, the model begins to be trained. After the training is completed, the results shown in Figure 1 are obtained. After manually closing the first graphics window, the results in Figure 2 are obtained, which is a confusion matrix. Finally, after manually closing the second graphics window, the verification parameter values based on the independent verification data set will be output in the command line interface: Accuracy, Recall, Precesion, and F1 Score.  

The trained model file, named "model-HIV_1_M_SPBEnv.pth", will be saved in current directory.  

HIV-1-M-SPBEnv was trained by using the original dataset (Table 1) plus the synthetic dataset (Table 2).  HIV-1-M-SPBEnv's classifcation accuracy was 100% for the independent dataset (Table 3). Figure 1 shows the model performance during training.  
![Loss function curve during training](images/Figure_1.png)   
**Figure 1.** Illustration of Model performance. On the left, the figure delineates the relationship between the Total training loss and the Total test loss over a span of 100 epochs. Total train loss and Total test loss are the sums of the model reconstruction loss obtained by training the Autoencoder and the training loss of the classifier, based on the model training data set and the model test data set respectively. On the right, the figure shows the trajectory of classification accuracy during 100 epochs of training based on the test data set.    
![Confusion matrix](images/Figure_2.png)   
**Figure 2.** Confusion matrix based on the independent validation data set

## 6. Uninstall HIV-1-M-SPBEnv
````python
pip uninstall HIV-1-M-SPBEnv
````
## 7. Genetic operations for generating new env gene sequence samples
I adopted artificial molecular evolution methods for DNA sequence sample synthesis, including synonymous mutation, non-synonymous mutation, insertion mutation, deletion mutation and large fragment recombination. The complete codebase for artificial genetic operations is accessible at the following link: https://github.com/pengsihua2023/HIV-1-M-SPBEnv/tree/main/code  
## 8　How can I prove that the synthetic samples are reliable? 　　
### 8.1 By using unrooted phylogenetic tree  
I randomly selected 5 L synthetic samples from the 500 synthesized L samples, and jointly constructed a phylogenetic tree with 42 gold standard samples of 12 subtypes of HIV type 1. The results show (Figure 3) that the synthesized L samples and the gold standard L sample were completely clustered together. On the other hand, I will use synthetic samples and original samples together to form training samples for training to obtain a deep learning model. Finally, I use an independent test data set to verify the results again.　  
![synthetic Sample H Tree](images/image-L-5-synthetic.png)    
**Figure 3.** Verification of the usability of generated samples. Marked in green are the 5 synthetic L samples which clustered together with the original L samples (in red), showing that the synthetic samples are potential original L samples.  

### 8.2 By using t-SNE visualzation based on K-mer vectorization
The labeld DNA samples were visualized with t-distributed stochastic neighbor embedding (t-SNE) and each gene sequence
is represented as a dot in the 2D plot. The DNA data were vectorized by K-mer method.  
![synthetic Sample H Tree](images/Figure_4.png)    
**Figure 4.** Verification of the usability of generated samples by using t-SNE. One color represents a category, and each category has 89 samples, including original samples and synthetic samples.

## 9. Model deployment
I deployed a trained model on a dedicated server, which is publicly available at:  
http://www.peng-lab.org:5000/analysis, to make it easy for biomedical researcher to perform HIV 1 subtype classification in their research.  
Users can upload their env gene sequences of HIV to the server, and then they can quickly obtain the predicted results of the HIV 1 subtype classification.   
## 10. The data sets
### 10.1 The original dataset
The detailed composition of the DNA sequence data of HIV env gene is shown in Table 1. For this data set, the sample size of some subtypes is too small, and there are only 2-5 samples in 4 subtypes. It is obviously impossible to build a machine learning model on such data. Then the only way is to find ways to increase the sample size for the small sample size subtypes.  
### 10.2 The synthetic dataset
I adopted molecular evolution methods for DNA sequence sample synthesis, including synonymous mutation, non-synonymous mutation, insertion mutation, deletion mutation and env gene large fragment recombination. Synthetic data samples are included in both the model training dataset (Table 2) and the model evaluation dataset (Table 3).    
#### Table 1 The original data set of HIV-1 M group env DNA sequences.
| Subtype |Sample Size | Subtype |Sample Size |
|-------|---------|-------|---------|
| A1 | 317 | F2 |16 |
| A2 | 5 | G |136 |
| B | 2,885 | H |10 |
| C | 1,717 | J |5 |
| D | 145 | K |2|
| F1 | 73 | L |3 |
### 10.3 The Training dataset
#### Table 2 The dataset for model training.  
| Subtype |Sample Size | Subtype |Sample Size |
|-------|---------|-------|---------|
| A1 | 500 (300)* | F2 |500 (12) |
| A2 | 500 (4) | G |500 (120) |
| B | 500 (500) | H |500 (8) |
| C | 500 (500) | J |500 (4) |
| D | 500 (120) | K |500 (1)|
| F1 | 500 (60) | L |500 (2) |  

#### * 500 (300): The number in bracket is the original sample data. In this case the sample size of synthetic data set is 500-300=200.  
### 10.4 The independent evaluation dataset
#### Table 3 The Independent data set for model evaluation＊.
| Subtype |Sample Size | Subtype |Sample Size |
|-------|---------|-------|---------|
| A1 | 11 | F2 |4 |
| A2 | 1 | G |16 |
| B | 2,387 | H |2 |
| C | 1,217 | J |1 |
| D | 25 | K |1|
| F1 | 13 | L |1 |

＊All the independent testting dataset is from the original dataset. 
## 11. License
HIV-1-M-SPBEnv has a MIT license, as seen in the [LICENSE.md](https://github.com/pengsihua2023/HIV-1-M-SPBEnv/blob/main/LICENSE) file.  
## 12. Citation
Sihua Peng, Ming Zhang. HIV-1-M-SPBEnv: HIV-1 M group subtype prediction using convolutional Autoencoder and full connected neural network. (Manuscript under review)   
## 13. Contact
If you have any questions, please feel free to contact Sihua Peng (Email: Sihua.Peng@uga.edu)      

Pull requests are highly welcomed!   
