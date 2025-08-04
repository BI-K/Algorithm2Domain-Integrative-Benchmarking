# Evaluation Framework

## Work in progress
[Algorithm2Domain](https://github.com/BI-K/Algorithm2Domain) is  a meta-repository for benchmarking of domain adaptation performance managed by the [Institute for Biomedical Informatics](https://bik.uni-koeln.de/) (University Hospital of Cologne).
Our goal is to aggregate existing algorithms and benchmarking suits, and develop integrative pipelines for mix-and-match cross-domain benchmarking. The data sources suitable for the benchmarking will be aggregated as pointers. 

This repository will contain a integrative benchmarking suite, that connects the datasets, models, domain adaptation algorithms, few-shot approaches and datasets from various domain adaptation benchmarking suites. 
For now our seed benchmarking suite is ADATime by *by: Mohamed Ragab\*, Emadeldeen Eldele\*,  Wee Ling Tan, Chuan-Sheng Foo, Zhenghua Chen<sup>&#9768;</sup>, Min Wu, Chee Kwoh, Xiaoli Li*.


## Wiki
In the Wiki we aggregate knowledge regarding domain adaptation and related topics: generalizability, out-of-distribution performance etc., as well as different methods to improve the algorithm design and to efficiently adapt it to the new domain.

## Contacts
Contact persons: Mayra Elwes (Mayra.Elwes at uk-koeln.de, technical questions), Ekaterina Kutafina (ekaterina.kutafina at uni-koeln.de, collaborations, contributions, development).

## License
The Apache 2.0 license concerns the original code developed in this meta-repository. For sub-repositories and other resources please check the corresponding license agreements.

## Citation
TODO

## Setup

Install torch as required by your computers setup, for me it was:
```
torch==2.7.1+cu126
torchaudio==2.7.1+cu126
torchmetrics==1.8.0
torchvision==0.22.1+cu126
```

The install all other packages as specified in the requirements.txt:
```pip install -r requirements.txt```



## How to Start Evaluation Framework

From the command line:
```python main.py --phase train --dataset WEATHER --backbone CNN --da_method NO_ADAPT```
```python main.py --phase test --dataset WEATHER --backbone CNN --da_method NO_ADAPT```

From the command line for hyperparameter tuning and documentation of experiemns via [Weight&Biases](https://wandb.ai/):
```python main_sweep.py --dataset WEATHER --backbone CNN --da_method NO_ADAPT```

### With a User Interface 
```streamlit run integrative_benchmarking_app.py```