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
cd src_cora
python run_GNN.py --dataset Cora 
```

For example to run for Citeseer with random splits:
```
cd src_citeseer
python run_GNN.py --dataset Citeseer 
```

For example to run for Pubmed with random splits:
```
cd src_pubmed
python run_GNN.py --dataset Pubmed 
```
For example to run for Computers with random splits:
```
cd src_computers
python run_GNN.py --dataset Computers
```
For example to run for Photo with random splits:
```
cd src_photo
python run_GNN.py --dataset Photo 
```
For example to run for ogbn-arxiv with random splits:
```
cd src_ogbnarxiv
python run_GNN.py --dataset ogbn-arxiv 
```
## Running DropEdge
To run the experiments of IncepGCN on the datesets, you can reproduce by the scripts in /script/supervised/
For the result of IncepGCN-DropEdge on the cora dataset: sh /script/supervised/cora_IncepGCN.sh
For the result of IncepGCN on the cora dataset: sh /script/supervised/cora_IncepGCN_nodrop.sh
For the result of IncepGCN-DropEdge on the citeseer dataset: sh /script/supervised/citeseer_IncepGCN.sh
For the result of IncepGCN on the citeseer dataset: sh /script/supervised/citeseer_IncepGCN_nodrop.sh
For the result of IncepGCN-DropEdge on the pubmed dataset: sh /script/supervised/pubmed_IncepGCN.sh
For the result of IncepGCN on the pubmed dataset: sh /script/supervised/pubmed_IncepGCN_nodrop.sh
For the result of IncepGCN-DropEdge on the Photo dataset: sh /script/supervised/photo_IncepGCN.sh
For the result of IncepGCN on the Photo dataset: sh /script/supervised/photo_IncepGCN_nodrop.sh
For the result of IncepGCN-DropEdge on the Computers dataset: sh /script/supervised/computers_IncepGCN.sh
For the result of IncepGCN on the Computers dataset: sh /script/supervised/computers_IncepGCN_nodrop.sh

## Code Structure
    src/: Model definition and training code.
    
    data/: Example datasets.
    

