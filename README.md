# FGND

## Running the experiments

### Requirements
Dependencies (with python >= 3.8):
Main dependencies are
torch==1.11
torch-cluster==1.5.9
torch-geometric==1.7.0
torch-scatter==2.0.6
torch-sparse==0.6.9
torch-spline-conv==1.2.1
torchdiffeq==0.2.1
Commands to install all the dependencies in a new conda environment
```
conda create --name FGND python=3.8
conda activate FGND

pip install ogb pykeops
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torchdiffeq -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-geometric
```

### Troubleshooting

There is a bug in pandas==1.3.1 that could produce the error ImportError: cannot import name 'DtypeObj' from 'pandas._typing'
If encountered, then the fix is 
pip install pandas==1.3.0 -U


### Dataset and Preprocessing
create a root level folder
```
./data
```
This will be automatically populated the first time each experiment is run.



## Running examples
For example to run for Cora with random splits:
```
cd src
python run_GNN.py --dataset Cora 
```


## Code Structure
    src/: Model definition and training code.
    
    data/: Example datasets.
    

