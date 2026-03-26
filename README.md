# Probing SRL Model Capabilities with Checklist-Style Challenge Tests
### Keze Hu (2902283)
This is the repository for the take-home assignment of the course Advanced NLP1 at Vrije Universiteit Amsterdam.
## Quick Set-ups
**Please do not change the repository structures when running the notebooks. They use relative paths to map to files.**
### Downloading Models
In this work, a trained feature-based (logreg) and a transformer-based (bert) model on Universal PropBank are used. They are available for direct download using the links below.
Please follow the installation instructions. Access to these models requires a VU account. For external users, please reach out to k.hu3@student.vu.nl.
#### Logistic regression
- Link: https://vunl-my.sharepoint.com/:f:/g/personal/k_hu3_student_vu_nl/IgD682i2uiUQSaLibmYlROOJAecA3_t8W4ctGMTXXqS95gU?e=19AFAm
- The two files should be placed in the folder `Models`). You also need a VU account to access OneDrive.
#### BERT
- Link: [https://vunl-my.sharepoint.com/:f:/g/personal/k_hu3_student_vu_nl/IgD682i2uiUQSaLibmYlROOJAecA3_t8W4ctGMTXXqS95gU?e=19AFAm](https://vunl-my.sharepoint.com/:u:/g/personal/k_hu3_student_vu_nl/IQAL_SRJEcA5QZjPgraU0p-QAfiO-A9Jh6sz-cjQssoEklY?e=E9Zc8S)
- This zip file should be unzipped and placed in the folder `Models`.

### Installing libraries
You might also need to install libraries in the `requirements.txt` file. Run the following code in the terminal:

`pip install -r requirements.txt`

The spaCy parser `en-core-web-lg` is also used. You need to download it before running the codes. Run the following code in the terminal as well:

`python -m spacy download en_core_web_lg`

## Repository Structures
###  \ `logreg_cha_srl.ipynb`
This is the first major notebook that runs the logistic regression on the challenge dataset. It uses scripts `a1standalone.py` and `a1calculation.py`. You will need to import them by executing the first line.
To make this work, model files should be placed in `Models`.
###  \ `Bert_cha_srl.ipynb`
This is the second major notebook that runs fine-tuned BERT on the challenge dataset. It uses scripts `a2standalone.py` and `a2calculation.py`. You will need to import them by executing the first line.
To make this work, the model folder `bertsrl` should be placed in `Models`.
### \ `Data`
This is the challenge dataset folder contain eight `.tsv` files. They are uniform in formats with four columns:
1) use cases
2) predicate;token in focus
3) predicate labels (separated by commas)
4) argument labels (separated by commas)
### \ `Generation_prompts_codes`
This folder documents the engineered prompts and raw generation of Gemini 3.1 Pro.

It also contains `B1+E1_DIR_predsub_pp,ipynb` used for mask-filling generation methods. To run it, make sure you install `requirements.txt`.
### \ `Predictions`
It contains model's output (in the last column) for the challenge tests.
