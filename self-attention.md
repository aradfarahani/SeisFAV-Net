$$
\begin{align}
&\textbf{REQUIRE: } \text{Input feature map } X \in \mathbb{R}^{C \times N}, \text{ reduction factor } r = 8, \\
&\qquad\qquad \text{learnable weights } W_Q, W_K \in \mathbb{R}^{(C/8) \times C}, W_V \in \mathbb{R}^{C \times C} \\
&\textbf{ENSURE: } \text{Output feature map } X' \text{ with enhanced long-range dependencies} \\
\\
&\textbf{PROJECTION COMPUTATION:} \\
&1: \quad Q \leftarrow W_Q X \quad \triangleright \text{ Query projection via } 1 \times 1 \text{ convolution (Equation 6)} \\
&2: \quad K \leftarrow W_K X \quad \triangleright \text{ Key projection via } 1 \times 1 \text{ convolution (Equation 6)} \\
&3: \quad V \leftarrow W_V X \quad \triangleright \text{ Value projection via } 1 \times 1 \text{ convolution (Equation 6)} \\
&4: \quad d_k \leftarrow C/8 \quad \triangleright \text{ Reduced dimension for computational efficiency} \\
\\
&\textbf{ATTENTION WEIGHT CALCULATION:} \\
&5: \quad S \leftarrow Q K^T \quad \triangleright \text{ Compute similarity scores (dot product)} \\
&6: \quad S \leftarrow S / \sqrt{d_k} \quad \triangleright \text{ Scale by } \sqrt{d_k} \text{ to stabilize gradients (Equation 7)} \\
&7: \quad \textbf{for } \text{each position } i \text{ in } S \textbf{ do} \quad \triangleright \text{ Apply softmax normalization (Equation 8)} \\
&8: \quad\quad A_i \leftarrow \frac{\exp(S_i)}{\sum_j \exp(S_j)} \quad \triangleright \text{ Softmax: } A = \text{softmax}(QK^T / \sqrt{d_k}) \\
&9: \quad \textbf{end for} \\
\\
&\textbf{FEATURE AGGREGATION WITH RESIDUAL CONNECTION:} \\
&10: \quad X_{\text{att}} \leftarrow V A^T \quad \triangleright \text{ Weighted aggregation of value features} \\
&11: \quad X' \leftarrow X_{\text{att}} + X \quad \triangleright \text{ Add residual connection (Equation 9)} \\
&12: \quad \textbf{return } X' \quad \triangleright \text{ Output with enhanced temporal coherence}
\end{align}
$$
