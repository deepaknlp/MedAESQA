# MedAESQA: Evidence-supported answers to medical questions



Please install [Anaconda](https://www.anaconda.com/distribution/) to create a conda environment as follow:
```shell script
# preparing environment
conda env create -f environment.yml
conda activate medaesqa
```

## Data Preparation
1) Download the MedAESQA dataset from [OSF repository](https://doi.org/10.17605/OSF.IO/ydbzq) and place medaesqa_v1.json in `data` directory




## Running Evaluation

```shell script
export PYTHONPATH=$PYTHONPATH/path/to/the/medaesqa/src directory
python medaesqa_eval.py
```

