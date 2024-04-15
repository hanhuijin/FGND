Reply:
One significant cause of oversmoothing is the presence of numerous erroneous connections in the graph topology, which leads to a reduction in node embeddings when aggregating information. These noises propagate between layers, resulting in oversmoothing issues.

The theoretical foundation for combating overfitting using partial differential equations and graph diffusion methods based on node embeddings lies in their effectiveness in filtering out noise in the graph topology. This prevents misaggregation of node information due to erroneous inter-class connections and facilitates better aggregation of neighbor node information. Moreover, the numerical solution methods for graph diffusion processes also include residual connections similar to those in ResNet.

Taking the following formula as an example, in GCN, the aggregation of neighboring nodes is equivalent to
$$
\mathbf{X}(T+1) = \Delta T \cdot \frac{\partial}{\partial t} \mathbf{x}(T).
$$
While the explicit Euler method, as the simplest form of PDE-based approach, adds residual connections on top of GCN:
$$
\mathbf{X}(T+1) = \mathbf{X}(T) + \Delta T \cdot \frac{\partial}{\partial t} \mathbf{x}(T).
$$
For more complex Runge-Kutta numerical integration methods:
$$
    \mathbf{K1}= \Delta T\cdot\frac{\partial}{\partial t} \mathbf{x}(T)= \Delta T\cdot\overline{\mathbf{A}}(\mathbf{x}(T))\mathbf{x}(T),\\
    \mathbf{K2}= \Delta T\cdot\frac{\partial}{\partial t} \mathbf{x}\left(T+\frac{\Delta T}{2} \right)= \Delta T\cdot\overline{\mathbf{A}}\left(\mathbf{x}\left(T+\frac{\Delta T}{2}\right)\right)\mathbf{x}\left(T+\frac{\Delta T}{2}\right),\\
    \mathbf{K3}= \Delta T\cdot\frac{\partial}{\partial t} \mathbf{x}\left(T+\frac{\Delta T}{2} \right)= \Delta T\cdot\overline{\mathbf{A}}\left(\mathbf{x}\left(T+\frac{\Delta T}{2}\right)\right)\mathbf{x}\left(T+\frac{\Delta T}{2}\right),\\
    \mathbf{K4}= \Delta T\cdot\frac{\partial}{\partial t} \mathbf{x}(T+\Delta T)= \Delta T\cdot\overline{\mathbf{A}}(\mathbf{x}(T+\Delta T))\mathbf{x}(T+\Delta T),\\
    \mathbf{X}(T+1)=\mathbf{X}(T)+\frac{1}{6}\cdot(\mathbf{K1}+2\mathbf{K2}+2\mathbf{K3}+\mathbf{K4}).
$$
This can be understood as handling temporal models in a manner similar to long short-term memory networks, using high-precision numerical solution methods to better aggregate information from different order neighbors. This allows the model to learn local information more effectively, capturing more multi-hop neighbor information while still remembering local details as layers stack.

It's worth noting that in FGND, we distinguish the aggregation weights for different adjacency relationships. If too much noise information is incorrectly aggregated in one layer, it will gradually accumulate with the increase in layers, eventually leading to oversmoothing. We have also supplemented related analyses in [Anonymized Repository - Anonymous GitHub (4open.science)](https://anonymous.4open.science/r/fgnd/More Theoretical Analysis For Parameter.md), demonstrating that our model can effectively assign different weights to intra-class and inter-class connections, thereby filtering out this part of noise.



Reply:
We plan to supplement a comprehensive discussion about the over-smoothing issue in related work as follows:

Graph Neural Network (GNN) models suffer from significant performance degradation across various graph learning tasks as the depth of GNNs increases. This performance decline is widely attributed to the problem of oversmoothing [1, 2, 3]. Intuitively, GNNs aggregate neighborhood information to generate new node representations, ultimately using these representations to differentiate between different nodes. With increasing depth, node representations become indistinguishable, leading to oversmoothing. This is because a large amount of noise exists in the adjacency relationships, which propagates between layers during the aggregation of neighborhood information. Several algorithms have been proposed to alleviate oversmoothing in GNNs, including skip connections and dilated convolutions [4], JumpKnowledge [5], DropEdge [6], PairNorm [7], Graph Neural Diffusion (GRAND) [8], GRAPH NEURAL DIFFUSION WITH A SOURCE TERM (GRAND++) [9], and wave equation motivated GNNs [10].

[1]Qimai Li, Zhichao Han, and Xiao-Ming Wu. Deeper insights into graph convolutional net-
works for semi-supervised learning. In Thirty-Second AAAI conference on artificial intelli-
gence, 2018.
[2]Kenta Oono and Taiji Suzuki. Graph neural networks exponentially lose expressive power for
node classification. In International Conference on Learning Representations, 2020.
[3]Deli Chen, Yankai Lin, Wei Li, Peng Li, Jie Zhou, and Xu Sun. Measuring and relieving the
over-smoothing problem for graph neural networks from the topological view. In Proceedings
ofthe AAAI Conference on Artificial Intelligence, volume 34, pages 3438–3445, 2020.
[4]Guohao Li, Matthias Muller, Ali Thabet, and Bernard Ghanem. Deepgcns: Can gcns go as
deep as cnns? In Proceedings ofthe IEEE/CVF International Conference on Computer Vision,
pages 9267–9276, 2019.
[5]Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, and
Stefanie Jegelka. Representation learning on graphs with jumping knowledge networks. In
International Conference on Machine Learning, pages 5453–5462. PMLR, 2018.
[6]Yu Rong, Wenbing Huang, Tingyang Xu, and Junzhou Huang. Dropedge: Towards deep
graph convolutional networks on node classification. In International Conference on Learning
Representations, 2020.
[7]Lingxiao Zhao and Leman Akoglu. PairNorm: Tackling oversmoothing in GNNs. In Interna-
tional Conference on Learning Representations, 2020.
[8]Ben Chamberlain, James Rowbottom, Maria I Gorinova, Michael Bronstein, Stefan Webb,
and Emanuele Rossi. GRAND: Graph neural diffusion. In Marina Meila and Tong Zhang,
editors, Proceedings of the 38th International Conference on Machine Learning, volume 139
of Proceedings ofMachine Learning Research, pages 1407–1418. PMLR, 18–24 Jul 2021.
[9]Matthew Thorpe, Tan Minh Nguyen, Hedi Xia, Thomas Strohmer, Andrea
Bertozzi, Stanley Osher, and Bao Wang. 2022. GRAND++: Graph Neural Diffu-
sion with A Source Term. In International Conference on Learning Representations.
1–9.
[10]Moshe Eliasof, Eldad Haber, and Eran Treister. PDE-GCN: Novel architectures for graph
neural networks motivated by partial differential equations. arXiv preprint arXiv:2108.01938,
2021.