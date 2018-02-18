# Deep Learning with Exponential Linear Units (ELUs)

The paper ‘Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs) [1] by Djork-Arn ́e
Clevert, Thomas Unterthiner & Sepp Hochreiter introduces an activation function ELU which provides a significant speed up in
learning rate over traditionally used activation functions (ReLU, eReLU, lReLU) by alleviating the bias shift effect and pushing
the mean of the activation function to zero. In the experiments section the paper proposes,‘ELUs lead not only to faster learning,
but also to significantly better generalization performance than sReLU and lReLUs on networks with more than 5 layers’. In this
project we examine the mathematical expressions and properties of different activation functions and we try to reestablish the
results achieved in this paper by training a 6 Layer feed forward neural network classifier on MNSIT dataset using different
activation functions.

## Pre-requisite

Please see requirements.txt file for require packages


## How to Run

Run the driver file run.py as

`python run.py`

which trigger running whole project

We have also built a feed forward neural network from scratch to get understanding the concepts.
It's implemented in file fee_fwd_NN_from_scratch.py

## Authors
* Prashant Gonarkar (pgonarka@asu.edu) 
* Sagar Patni (shpatni@asu.edu)


## License
This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments 
* [Fast and accurate deep network learning by exponential linear units] (https://arxiv.org/abs/1511.07289)

