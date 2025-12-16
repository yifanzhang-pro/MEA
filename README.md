# MEA

**Matrix Exponential Attention**

**Author**: Yifan Zhang

**Date**: December 15, 2025

$$
\mathrm{MExp}(\mathbf{Q} \mathbf{K}^{\top}) \mathbf{V} \approx \sum_{k=0}^{H} \frac{1}{k!} \mathrm{HLA}_k(\mathbf{Q}, \mathbf{K}, \mathbf{V})
$$

MEA approximates the matrix exponential of attention scores via a truncated Taylor series. By leveraging the state-space realization of **Higher-order Linear Attention (HLA)**, MEA computes high-order interaction terms (powers of the attention matrix) in linear time without materializing $n \times n$ matrices.

See [Higher-order Linear Attention (HLA)](https://arxiv.org/abs/2510.27258) for the theoretical foundation of the streaming algorithms used here.

## Mathematical Formulation

Standard Scaled Dot-Product Attention utilizes the softmax nonlinearity:

$$
\mathrm{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}
$$

MEA replaces the softmax with the **Matrix Exponential** ($\mathrm{MExp}$). For an unnormalized attention matrix $\mathbf{A} = \mathbf{Q}\mathbf{K}^\top$:

$$
\mathrm{MExp}(\mathbf{A})\mathbf{V} = e^{\mathbf{A}}\mathbf{V} = \left( \sum_{k=0}^{\infty} \frac{1}{k!} \mathbf{A}^k \right) \mathbf{V}
$$

We approximate this by truncating the series at order $H$ (typically $H=2$).

### Recursive Decomposition via AHLA

Explicitly computing the matrix power $\mathbf{A}^k$ would require **$\mathcal{O}(n^3)$** complexity, and even an optimized iterative product $\mathbf{A}(\mathbf{A}\mathbf{V})$ scales as $\mathcal{O}(n^2)$.

MEA exploits the associativity of matrix multiplication to factorize these terms into streaming updates with **$\mathcal{O}(n)$ complexity**.

The expansion up to order $H=2$ is:

$$
\mathbf{Y} \approx \underbrace{\mathbf{V}}_{\text{Order 0}} + \underbrace{(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}}_{\text{Order 1}} + \frac{1}{2!} \underbrace{(\mathbf{Q}\mathbf{K}^\top)^2 \mathbf{V}}_{\text{Order 2}}
$$

#### Order 1: Linear Attention

The first-order term corresponds to standard Linear Attention ($k=1$):

$$
\mathbf{Y}_1 = \mathbf{Q} \left( \sum_{j} \mathbf{k}_j \mathbf{v}_j^\top \right)
$$

#### Order 2: Asymmetric HLA

The second-order term utilizes the AHLA operator. It represents the path $\mathbf{Q} \to \mathbf{K}^\top \to \mathbf{Q} \to \mathbf{K}^\top \to \mathbf{V}$:

$$
\mathbf{Y}_2 = \mathbf{Q} (\mathbf{K}^\top \mathbf{Q}) (\mathbf{K}^\top \mathbf{V})
$$

This admits the following streaming sufficient statistics (per head), ensuring $\mathcal{O}(1)$ memory cost per token:

$$
\begin{aligned}
\mathbf{P}_t^{KV} &= \sum_{j \le t} \mathbf{k}_j \mathbf{v}_j^\top &\in \mathbb{R}^{d \times d_v} \\
\mathbf{E}_t &= \sum_{i \le t} \mathbf{k}_i (\mathbf{q}_i^\top \mathbf{P}_i^{KV}) &\in \mathbb{R}^{d \times d_v}
\end{aligned}
$$

The output at time $t$ for the second-order term is:

$$
\mathbf{o}_t^{(2)} = \mathbf{q}_t^\top \mathbf{E}_t
$$

### Exact Causal Masking

To enforce strict autoregressive causality (masking the upper triangular of $\mathbf{Q}\mathbf{K}^\top$), MEA relies on the **Extended Summaries** theorem. For the symmetric interpretation of second-order interactions, we maintain cross-moment summaries $\mathbf{G}_t$ to subtract acausal contributions dynamically:

$$
\mathbf{o}_t^{\text{sym}} = \mathbf{q}_t^\top \left( \mathbf{S}_t^K \mathbf{C}_t^{QV} - \mathbf{G}_t \right)
$$

Where:

$$
\mathbf{S}_t^K = \sum_{i \le t} \mathbf{k}_i \mathbf{k}_i^\top
$$ 

$$
\mathbf{G}_t = \sum_{i \le t} (\mathbf{k}_i \mathbf{k}_i^\top) \mathbf{C}_{i-1}^{QV}
$$

This ensures the model is mathematically equivalent to a masked Transformer while maintaining the efficiency of an RNN.

## Citation

```bibtex
@article{zhang2025matrix,
   title   = {Matrix Exponential Attention},
   author  = {Zhang, Yifan},
   journal = {yifanzhang-pro.github.io},
   year = {2025},
   month = {December},
   url = "https://github.com/yifanzhang-pro/MEA"
}

@article{zhang2025hla,
   title   = {Higher-order Linear Attention},
   author  = {Zhang, Yifan and Qin, Zhen and Gu, Quanquan},
   journal = {arXiv preprint 2510.27258},
   year    = {2025}
}
```
