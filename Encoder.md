$$
\begin{align}
&\textbf{REQUIRE: } \text{Noisy input } x_{\text{noisy}} \in \mathbb{R}^{C \times N}, \text{ kernel size } k = 3, \text{ stride } s = 2, \\
&\qquad\qquad \text{encoder depth } L = 3, \text{ initial channels } C_0 \\
&\textbf{ENSURE: } \text{Hierarchical features } h_{\text{bottleneck}} \text{ and skip connections } \{e_1, e_2, e_3\} \\
\\
&\textbf{ENCODER (Hierarchical Feature Extraction):} \\
&1: \quad h_1 \leftarrow x_{\text{noisy}} \quad \triangleright \text{ Input: } N \text{ temporal samples} \\
&2: \quad \textbf{for } l = 1 \text{ to } L \textbf{ do} \quad \triangleright \text{ Three residual blocks with downsampling} \\
&3: \quad\quad h_l \leftarrow \text{Conv}(h_l, k=3) \to \text{BN} \to \text{ReLU} \quad \triangleright \text{ First convolutional layer} \\
&4: \quad\quad h_l \leftarrow \text{Conv}(h_l, k=3) \to \text{BN} \to \text{ReLU} \quad \triangleright \text{ Second convolutional layer (residual block)} \\
&5: \quad\quad \textbf{if } l \geq 2 \textbf{ then} \quad \triangleright \text{ Apply self-attention in EncBlock 2 and 3} \\
&6: \quad\quad\quad h_l \leftarrow \text{SelfAttention}(h_l) \quad \triangleright \text{ Self-attention module (Section C)} \\
&7: \quad\quad \textbf{end if} \\
&8: \quad\quad e_l \leftarrow h_l \quad \triangleright \text{ Store encoder features for skip connections} \\
&9: \quad\quad h_{l+1} \leftarrow \text{MaxPool}(h_l, \text{stride}=2) \quad \triangleright \text{ Downsample: } N \to N/2 \to N/4 \\
&10: \quad \textbf{end for} \\
&11: \quad h_{\text{bottleneck}} \leftarrow h_{L+1} \quad \triangleright \text{ Bottleneck features at resolution } N/2^L \\
&12: \quad \textbf{return } h_{\text{bottleneck}}, \{e_1, e_2, e_3\}
\end{align}
$$
