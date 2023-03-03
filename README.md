# Step-oriented Graph Convolutional Networks in Representation Learning
This repository contains the PyTorch implementation code for the paper "Step-oriented Graph Convolutional Networks in Representation Learning"

## Dependencies
- CUDA 11.0
- python 3.10.8
- pytorch 1.13.0
- torch-geometric 2.1.0
- torchmetrics 0.11.0
- numpy 1.23.5
- tqdm 4.64.1

## Datasets
We used four benchmark datasets; Cora, CiteSeer, PubMed, and Flickr. The [data/](https://github.com/dxlabskku/IJCAI_StepGCN/tree/main/data) folder contains the Cora benchmark dataset. You can refer to torch-geometric documentation to use other datasets [here](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html).

## Results
Testing accuracy are summarized below.

<table>
  <tr>
    <td><b>Dataset</b></td>
    <td><b>GRCN Type</b></td>
    <td><b>ResBlock Type</b></td>
    <td align="right"><b>Accuracy</b></td>
    <td align="right"><b>Depth & Step</b></td>
  </tr>
  <tr>
    <td rowspan="2">Cora</td>
    <td rowspan="2">graph-div.</td>
    <td>-</td>
    <td align="right">0.808</td>
    <td align="right">2-layer</td>
  </tr>
  <tr>
    <td>none</td>
    <td align="right">0.809</td>
    <td align="right">1-step</td>
  </tr>
  <tr>
    <td rowspan="2">CiteSeer</td>
    <td rowspan="2">conv.</td>
    <td>-</td>
    <td align="right">0.700</td>
    <td align="right">2-layer</td>
  </tr>
  <tr>
    <td>none</td>
    <td align="right">0.701</td>
    <td align="right">7-step</td>
  </tr>
  <tr>
    <td rowspan="2">PubMed</td>
    <td rowspan="2">none</td>
    <td>-</td>
    <td align="right">0.790</td>
    <td align="right">2-layer</td>
  </tr>
  <tr>
    <td>none</td>
    <td align="right">0.792</td>
    <td align="right">2-step</td>
  </tr>
  <tr>
    <td rowspan="2">Flickr</td>
    <td rowspan="2">conv.</td>
    <td>-</td>
    <td align="right">0.511</td>
    <td align="right">10-layer</td>
  </tr>
  <tr>
    <td>linear</td>
    <td align="right">0.519</td>
    <td align="right">1-step</td>
  </tr>
</table>


## Usage
```
python train.py
```

You can run [train.py](https://github.com/dxlabskku/StepGCN/blob/main/train.py) with changed setting of arguments as follows:

<table>
  <tr>
    <td><b>Name</b></td>
    <td><b>Type</b></td>
    <td align="right"><b>Default</b></td>
  </tr>
  <tr>
    <td>grcn</td>
    <td>str</td>
    <td align="right">none</td>
  </tr>
  <tr>
    <td>depth</td>
    <td>int</td>
    <td align="right">2</td>
  </tr>
  <tr>
    <td>resblock</td>
    <td>str</td>
    <td align="right">none</td>
  </tr>
  <tr>
    <td>step</td>
    <td>int</td>
    <td align="right">0</td>
  </tr>
  <tr>
    <td>nhid</td>
    <td>int</td>
    <td align="right">16</td>
  </tr>
  <tr>
    <td>dataset</td>
    <td>str</td>
    <td align="right">cora</td>
  </tr>
  <tr>
    <td>seed</td>
    <td>int</td>
    <td align="right">42</td>
  </tr>
  <tr>
    <td>epochs</td>
    <td>int</td>
    <td align="right">1000</td>
  </tr>
  <tr>
    <td>paitence</td>
    <td>int</td>
    <td align="right">100</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>float</td>
    <td align="right">1e-2</td>
  </tr>
  <tr>
    <td>weight_decay</td>
    <td>float</td>
    <td align="right">5e-4</td>
  </tr>
  <tr>
    <td>dropout</td>
    <td>float</td>
    <td align="right">0.5</td>
  </tr>
</table>

Since there are 9 types of GRCNs and 4 types of ResBlocks, you can clarify the type of model you want to train by varying the argument "grcn" and "resblock".

Types of GRCNs are as follows:

<img src=https://user-images.githubusercontent.com/96400041/190964732-5b639e53-3487-4a58-8269-37322fb5af2b.jpg width="70%" height="50%"/>

normalized adjacency matrix can be multiplied to residual connections of every type to form graph-GRCN types.

Types of ResBlocks are as follows:

<img src=https://user-images.githubusercontent.com/96400041/190964803-7413d3d0-ad51-48de-9b0f-f992b69c7018.jpg width="50%" height="30%"/>

Therefore, options of "grcn" can be one of,

<table>
  <tr>
    <td>none</td>
    <td>seq</td>
    <td>lin</td>
    <td>div</td>
    <td>conv</td>
    <td>graph_seq</td>
    <td>graph_lin</td>
    <td>graph_div</td>
    <td>graph_conv</td>
  </tr>
</table>

and "resblock" can be one of,

<table>
  <tr>
    <td>none</td>
    <td>linear</td>
    <td>graph</td>
    <td>graph_linear</td>
  </tr>
</table>

## Citation
