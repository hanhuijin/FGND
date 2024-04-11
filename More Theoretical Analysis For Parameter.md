### Statistical Values for Edge Weights in Reconstructed Topology Structure

| Dataset    | Mean   | Min     | Max  | Standard Deviation | Variance | Range   |
| ---------- | ------ | ------- | ---- | ------------------ | -------- | ------- |
| CoauthorCS | 0.8584 | 0.0434  | 1    | 0.1943             | 0.0377   | 0.9566  |
| Citeseer   | 0.8008 | 0.0526  | 1    | 0.2502             | 0.0626   | 0.9474  |
| Pubmed     | 0.8646 | 0.0212  | 1    | 0.217              | 0.0471   | 0.9788  |
| Cora       | 0.862  | 0.125   | 1    | 0.2146             | 0.046    | 0.875   |
| Photo      | 0.8609 | 0.0111  | 1    | 0.2054             | 0.0421   | 0.9889  |
| Computers  | 0.8168 | 0.0119  | 1    | 0.2156             | 0.0465   | 0.9881  |
| Texas      | 0.4029 | 0.02386 | 1    | 0.1838             | 0.0338   | 0.97614 |
| Squirrel   | 0.4376 | 0.0897  | 1    | 0.2098             | 0.044    | 0.9103  |
| ogbn-arxiv | 0.707  | 0.0017  | 1    | 0.2718             | 0.00739  | 0.9983  |


The statistical values in the table represent the weights of edges in the reconstructed topology structure. From the table, it can be observed that reconstructing the topological relationships based on node latent representations effectively differentiates the original graph's topology. Combined with Table 3 in paper, it is evident that reconstructing the topological relationships based on node latent representations significantly reduces the weights of inter-class connections and enhances the weights of intra-class connections.
It's worth noting that the computation of $sim$ is based on node latent representations rather than solely on node features. In other words, the results of $sim$ are also learnable.

In fact, β can be understood as the sensitivity of γ to "sim". By cross-referencing Table 3 and Figure 3, we can observe that the optimal value of β is correlated with the properties of different datasets. We can determine the value of β within a rough range based on the characteristics of different datasets.

 **The role of parameter $\gamma$:**
 According to the equation $\mathbf{X}(T+1) = (1-\gamma )\mathbf{X}(T)+\gamma\int_{T}^{T+1} \frac{\partial \mathbf{X}(t)}{\partial t}{by D}d t$, we can see that $\gamma$ controls the weighted balance between the current node features and neighboring node features. When $\gamma$ approaches 0, the model tends to preserve the original node features, i.e., the influence of $(1-\gamma )\mathbf{X}(T)$ is greater. Conversely, when $\gamma$ approaches 1, the model tends to aggregate neighboring node features, i.e., the influence of $\gamma\int{T}^{T+1} \frac{\partial \mathbf{X}(t)}{\partial t}_{by D}d t$ is greater. Therefore, adjusting $\gamma$ can control the relative contributions of node features and neighboring node features during the information aggregation process, thereby optimizing the information propagation effect.

**The role of parameter $\beta$:** 
According to the equation $\gamma =(1-\beta)\cdot sim +\beta$, we can see that $\beta$ is used to adjust the dependence of parameter $\gamma$ on the similarity measure $sim$. When $\beta$ is small, the model relies more on the similarity measure $sim$, i.e., the value of $\gamma$ is mainly determined by the similarity of node features. Conversely, when $\beta$ is large, the model is more independent of the similarity measure $sim$, i.e., the value of $\gamma$ is mainly determined by $\beta$. Therefore, adjusting $\beta$ can control the model's dependence on the similarity measure during the information aggregation process, adapting to different datasets and task requirements.