# Monotone-Value Neural Networks: Exploiting Preference Monotonicity in Combinatorial Assignment

Published at [IJCAI 2022](https://www.ijcai.org/proceedings/2022/77)
This is a piece of software used for computing the prediction performance experiments shown in Table 1 and the MLCA efficiency experiments shown in Table 2 of the paper
[Monotone-Value Neural Networks: Exploiting Preference Monotonicity in Combinatorial Assignment](https://arxiv.org/abs/2109.15117). The algorithms are described in detail in the this paper.


## Requirements

* Python 3.7
* Java 8 (or later)
  * Java environment variables set as described [here](https://pyjnius.readthedocs.io/en/stable/installation.html#installation)
* JAR-files ready (they should already be)
  * CPLEX (>=12.10.0): The file cplex.jar (for 12.10.0) is provided in the folder lib.
  * [SATS](http://spectrumauctions.org/) (>=0.7.0): The file sats-0.7.0.jar is provided in the folder lib.

## Dependencies

Prepare your python environment (whether you do that with `conda`, `virtualenv`, etc.) and enter this environment.

Using pip:
```bash
$ pip install -r requirements.txt
```

* CPLEX Python API installed as described [here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html)
* Make sure that your version of CPLEX is compatible with the cplex.jar file in the folder lib.


## How to run

### Prediction Performance of MVNN vs. NN
First collect some data from an auction domain, e.g. 1 instance of GSVM:
```bash
$ python all_bids_generator.py --domain=GSVM --number_of_instances=1
```

| Parameter        | Explanation | Example  |Can be empty  |
| ------------- |-------------| -----|-----|
| domain      | SATS domain to choose | GSVM / LSVM / SRVM / MRVM | No |
| number_of_instances      | Number of instances of the SATS domain to save as training data | 5 | No |

The data is saved under data/GSVM/GSVM_seed1_all_bids.pkl

To run the prediction performance experiment from Table 1 of the main paper using the data you just collected do the following.
```bash
$ python simulation_prediction_performance.py --domain=GSVM --T=20 --bidder_type=national --network_type=MVNN --seed=1
```
| Parameter        | Explanation | Example  |Can be empty  |
| ------------- |-------------| -----|-----|
| domain      | SATS domain to choose | GSVM / LSVM / SRVM / MRVM | No |
| T      | Number of training data points | 10 | No |
| bidder_type      | Bidder type to choose | regional/national/local/high_frequency | No |
| network_type      | Whether to use MVNN or NN | MVNN/NN | No |
| seed      | Auction instance from which the training data was collected | 1 | No |

This script selects the winning configurations of the HPO on the prediction performance saved in prediction_performance_hpo_results.json.
Finally it prints the train/val/test metrics shown in the table and plots a true-predicted scatter plot of the test data.


### MLCA Experiments
This step requires cplex so make sure everything is setup as described above.
```bash
$ python simulation_mlca.py --domain=GSVM --network_type=MVNN
```
| Parameter        | Explanation | Example  |Can be empty  |
| ------------- |-------------| -----|-----|
| domain      | SATS domain to choose | GSVM / LSVM / SRVM / MRVM | No |
| network_type      | Whether to use MVNN or NN | MVNN/NN | No |
| seed      | Auction instance seed was collected | 1 | No |

## Acknowledgements

The MLCA and the MIP formulation of the Plain (ReLU) Neural Network is based on Weissteiner et al.[1]

[1] Weissteiner, Jakob, and Sven Seuken. "Deep Learning—Powered Iterative Combinatorial Auctions." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 02. 2020.

## Contact

Maintained by Jakob Weissteiner (weissteiner), Jakob Heiss (JakobHeiss) and Julien Siems (Julien)
