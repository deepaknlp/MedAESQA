# MedAESQA: Evidence-supported answers to medical questions



Please install [Anaconda](https://www.anaconda.com/distribution/) to create a conda environment as follows:
```shell script
# preparing environment
conda env create -f environment.yml
conda activate medaesqa
```
## Data Preparation
Download the MedAESQA dataset from [OSF repository](https://osf.io/ydbzq/) and place `medaesqa_v1.json` in `data` directory



## Running Evaluation on MedAESQA

```shell script
export PYTHONPATH=$PYTHONPATH/path/to/the/medaesqa/src directory
python medaesqa_eval.py
```


## Running Evaluation on BioGen 2024 topics
Obtain the Biogen assessment file and cluster files from TREC ([https://pages.nist.gov/trec-browser/trec33/biogen/data/]).

```shell script
export PYTHONPATH=$PYTHONPATH/path/to/the/medaesqa/src directory
python medaesqa_eval.py --path_to_processed_annotation_file <PATH_TO_BIOGEN_ASSESSMENT> --path_to_ST_cluster_file <PATH_TO_ST_CLUSTER_FILE> --path_to_SimCSE_cluster_file <PATH_TO_SIMCSE_CLUSTER_FILE>
```

## References

If you are using this code or dataset for your research work, please cite our papers:
```
@article{gupta2025dataset,
  title={A Dataset of Medical Questions Paired with Automatically Generated Answers and Evidence-supported References},
  author={Gupta, Deepak and Bartels, Davis and Demner-Fushman, Dina},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={1035},
  year={2025}
}

@article{guptaoverview,
  title={Overview of TREC 2024 Biomedical Generative Retrieval (BioGen) Track},
  author={Gupta, Deepak and Demner-Fushman, Dina and Hersh, William and Bedrick, Steven and Roberts, Kirk},
  journal={TREC},
  year={2024}
}
```
