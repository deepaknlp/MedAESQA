# MedAESQA: Evidence-supported answers to medical questions



Please install [Anaconda](https://www.anaconda.com/distribution/) to create a conda environment as follows:
```shell script
# preparing environment
conda env create -f environment.yml
conda activate medaesqa
```



## Running Evaluation on MedAESQA

```shell script
export PYTHONPATH=$PYTHONPATH/path/to/the/medaesqa/src directory
python medaesqa_eval.py
```


## Running Evaluation on BioGen 2024 top
Get the biogen assessment file from the TREC.

```shell script
export PYTHONPATH=$PYTHONPATH/path/to/the/medaesqa/src directory
python medaesqa_eval.py --path_to_processed_annotation_file <PATH_TO_BIOGEN_ASSESSMENT>
```
