# SQuAD

## Introduction

NLP system trained on Stanford Question Answering Dataset (SQuAD). SQuAD tests the ability of a system to not only answer reading comprehension questions, but also abstain when presented with a question that cannot be answered based on the provided paragraph.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### PREREQUISITES

Below you can find the instructions for downloading the dataset, the pre-trained embeddings and the weights of our trained model.

<!-- **Please note that the software will automatically download all of them, so you are not required to specifically follow the below procedure.** -->
<h3><b>Please note that the software will automatically download all of them, so you are not required to specifically follow the below procedure.</b></h3>

#### Optional Manual Procedure
First of all you have to create a new folder `/data` at the root level of the project.

The original dataset is available at the following [link](https://drive.google.com/file/d/19byT_6Hhx4Di1pzbd6bmxQ8sKwCSPhqg/view?usp=sharing). 
Once you have downloaded it, you have to place the unzipped file `training_set.json` inside the `/data/raw/` folder.

The pre-trained Glove embeddings used are available at the following [link](https://drive.google.com/file/d/15mTrPUQ4PAxfepzmRZfXNeKOJ3AubXrJ/view).
Once you have downloaded it, you have to place the unzipped file `Glove_50.txt` inside the `/data/raw/` folder.

The weights of the trained model are available at the following [link](https://drive.google.com/file/d/1mLDlYEV5kVY8vIqyQwm31AGpEs37-1fv/view?usp=sharing).
Once you have downloaded it, you have to place the unzipped file `DRQA.h5` inside the `/data/checkpoints/` folder.

#### Apple M1

If you are using the new Apple M1 chip please be sure to have installed `hdf5` by running:
```
$ brew install hdf5
```

### INSTALLING

below you can find all the scripts for installing based on your OS/processor
```
$ make
    > "+------------------------------------------------------+"
    > "|         OS         |  Hardware  |    Setup Command   |"
    > "+------------------------------------------------------+"
    > "|   Windows/Linux    |   - GPU    |  'make setup.CPU'  |"
    > "|   Windows/Linux    |   + GPU    |  'make setup.GPU'  |"
    > "|    Apple macOS     |    + M1    |  'make setup.M1'   |"
    > "|    Apple macOS     |    - M1    |  'make setup.CPU'  |"
    > "+------------------------------------------------------+"
```

for instance if you have MacOS with Intel chip you have to run:
```
$ make setup.CPU
```

or alternatively you can find all the different version of the `requirements` inside the `/tools/requirements` folder.

## Running the Tests

#### Training
You can train the model from scratch using your **custom** dataset by running:
```
$ python src/train.py "<path_of_your_json_dataset>"
```

#### Inference
You can run the inference procedure on a specific dataset by running:
```
$ python src/compute_answers.py "<path_of_your_json_dataset>"
```
once you have done this, you can see the output generated inside the `/data/predictions/answers.pred.json` file.

#### Evaluations
You can evaluate the performances of the inference by running:
```
$ python src/evaluate.py "<path_of_your_json_dataset>" data/predictions/answers.pred.json
```


<!-- ## Built With
* [Tensorflow](https://github.com/tensorflow/tensorflow) - Open Source ML Framework
* [Rosetta](https://github.com/acmeism/RosettaCodeData) - RosettaCode Dataset 
-->


## Authors

* Matteo Conti - *author* - [contimatteo](https://github.com/contimatteo)
* Francesco Palmisano - *author* - [Frankgamer97](https://github.com/Frankgamer97)
* Primiano Arminio Cristino - *author* - [primianocristino](https://github.com/primianocristino)
* Luciano Massaccesi - *author* - [fruscello](https://github.com/fruscello)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
