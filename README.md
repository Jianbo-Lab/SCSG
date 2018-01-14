# SCSG

Code for replicating the experiments in the paper [Non-Convex Finite-Sum Optimization Via SCSG Methods](https://papers.nips.cc/paper/6829-non-convex-finite-sum-optimization-via-scsg-methods.pdf) by Lihua Lei, Cheng Ju, Jianbo Chen, Michael I. Jordan. 

## Dependencies
This project runs with Python 2.7 and requires Tensorflow of version 1.2.1 or higher. Please `pip install` the following packages:
- `numpy`
- `tensorflow`
 
Or you may run the following command in shell to install the required packages:
```shell
git clone https://github.com/Jianbo-Lab/SCSG
cd SCSG
sudo pip install -r requirements.txt
```

## Running in Docker, MacOS or Ubuntu
We provide the source code to run the MNIST example. Run the following commands in shell:

```shell
###############################################
# Omit if already git cloned.
git clone https://github.com/Jianbo-Lab/SCSG
cd SCSG 
############################################### 
python experiments/run_mnist_experiment.py 
```

The configurations for each experiment can be changed in command-line interfaces. For example, the following code runs an MNIST experiment with a three-layer fully-connected neural network, SCSG for gradient update, fixed batchsize being 1,000, learning rate being 0.01 and the ratio of batch size and mini-batch size being 32 for 400 iterations: 

```shell 
python experiments/run_mnist_experiment.py --model 'fc' --method 'scsg' --batch_size 1000 --num_iterations 400 --learning_rate 0.01 --ratio 32 --fix_batch
```

See `experiments/run_mnist_experiment.py` for details. 
## Citation
If you use this code for your research, please cite our [paper](https://papers.nips.cc/paper/6829-non-convex-finite-sum-optimization-via-scsg-methods.pdf):
```
@inproceedings{lei2017non,
  title={Non-convex finite-sum optimization via scsg methods},
  author={Lei, Lihua and Ju, Cheng and Chen, Jianbo and Jordan, Michael I},
  booktitle={Advances in Neural Information Processing Systems},
  pages={2345--2355},
  year={2017}
}
```