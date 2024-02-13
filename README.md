# Spatial Graph INR
This model is an instance of implicit neural representations (INRs) for non-Euclidean data such as graphs.
Existing graph INRs, such as GINR, utilizes eigenvectors obtained from graph similarity metrices (spectral embeddings). Due to the nature of these metrices, they assume fixed graph structures and prone to even slight graph alternations. To resolve this problem, we have introduced a new graph INR, which takes spatial embeddings instead of spectral embeddings. We chose mixed-curvature product spaces composed of hyperbolic, spherical, and Euclidean spaces for embedding graphs with various structures.
Compared to GINR, spatial graph INR has shown improved results on graph signal prediction tasks on network-like graph data and graph super-resolution tasks on 3D-mesh data.




# Code References
Development of this model was based on the codes for the following projects:
Learning Mixed-Curvature Representations in Product Spaces: https://github.com/HazyResearch/hyperbolics
Generalised Impliced Neural Representations: https://github.com/danielegrattarola/GINR
