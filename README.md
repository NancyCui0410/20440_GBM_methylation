# 20440_GBM_methylation
This is the github repo for Yufei Cui and Kaiyi Jiang's 20.440 term projec: Utilizing DNA methylation data to predict glioblastoma diagnosis and unveil new therapeutic target. We used state of the art computational algorithms including support vector machine (SVM) and regularized linear regression to train a classifier based on publicly available whole-genome methylation data to predict GBM samples. Through this model, we hope to gain the ability to not only enhance the diagnosis of GBM but also unveil key epigenome-level transcription regulations responsible for the development and prognosis of GBM. 
## Installation
Packages: \
matplotlib==3.4.3 \
numpy==1.21.2 \
pandas==1.3.3 \
scikit_learn==1.0.2 \
scipy==1.7.1 \
seaborn==0.11.2
## Folder structure
The folders include first raw_data where preprocessed beta values of each datasets were uploaded. The files could be regenerated using first section of code in Methylation_GBM.py file. Follow the comment for more detailed instructions.\
There is an additional folder called 'Figures' where figures invovled in the final project writeup were uploaded. Individual images generated from the code were not included for storage savings.
## Data
DNA methylation data sets used in this project were acquired from GEO and TCGA. The datasets are TCGA-GBM, GSE15745, and GSE60274. Beta values of each samples were downloaded as .txt files. We did not reprocess the raw microarray data.\
## Running the methylation workflow
To reproduce the methylation analysis, clone this repo and modify the raw file locations as appropriate in the Methylation_GBM.py file according to the comment. Then run command from the folder
```
python Methylation_GBM.py
```
To see the figures generated by authors, open Methylation_GBM.ipynb and check the individual figures below each code block. This version is where all figures in the Figures folder came from
