# Basic INRs
python train_graph_inr.py --dataset_dir dataset/bunny_res2     --n_fourier 100 --n_nodes_in_sample 5000 --lr 0.0001 --n_layers 6 --skip=True --sine=True --all_sine=True
python train_graph_inr.py --dataset_dir dataset/protein_1AA7_A --n_fourier 100 --n_nodes_in_sample 5000 --lr 0.0001 --n_layers 6 --skip=True --sine=True --all_sine=True
python train_graph_inr.py --dataset_dir dataset/us_elections   --n_fourier 100 --n_nodes_in_sample 5000 --lr 0.0001 --n_layers 6 --skip=True --sine=True --all_sine=True

python train_graph_inr.py --model Spatial_Graph_INR --dataset_dir dataset/us_elections --emb_dir spatial_embeddings/US-county-fb/... --emb_name ... --n_nodes_in_sample 5000 --lr 0.0001 --n_layers 6 --skip=True --sine=True --all_sine=True
python train_graph_inr.py --model Spatial_Graph_INR --dataset_dir dataset/bunny_res2   --emb_dir spatial_embeddings/bunny_res2/...   --emb_name ... --n_nodes_in_sample 5000 --lr 0.0001 --n_layers 6 --skip=True --sine=True --all_sine=True

# Prediction tests
python train_graph_inr.py --dataset_dir dataset/us_elections   --mode pred --patience 2000 --lr_patience 500 --n_fourier 100 --n_nodes_in_sample -1 --lr 0.0001 --n_layers 6 --skip=True --sine=True --all_sine=True
python train_graph_inr.py --dataset_dir dataset/bunny_res2     --mode pred --patience 2000 --lr_patience 500 --n_fourier 100 --n_nodes_in_sample 5000 --lr 0.0001 --n_layers 6 --skip=True --sine=True --all_sine=True

python train_graph_inr.py --model Spatial_Graph_INR --dataset_dir dataset/us_elections --emb_dir spatial_embeddings/US-county-fb/... --emb_name ... --mode pred --patience 2000 --lr_patience 500 --n_nodes_in_sample -1 --lr 0.0001 --n_layers 6 --skip=True --sine=True --all_sine=True
python train_graph_inr.py --model Spatial_Graph_INR --dataset_dir dataset/bunny_res2   --emb_dir spatial_embeddings/bunny_res2/...   --emb_name ... --mode pred --patience 2000 --lr_patience 500 --n_nodes_in_sample 5000 --lr 0.0001 --n_layers 6 --skip=True --sine=True --all_sine=True

# Transferability: SBM
# Set max_epochs to 1000 because otherwise it takes too long and the performance
python train_graph_inr.py --n_fourier 3 --max_epochs 1000

# Transferability: super-resolution
# Changes to improve transferability:
#   - Use ReLU instead of sine
#   - Add a couple of extra layers (because of ReLU)
#   - Lower n_fourier to 7 (found empirically)
#   - Increase learning rate to 0.001
python train_graph_inr.py --dataset_dir dataset/bunny_res2 --n_fourier 7 --n_nodes_in_sample 5000 --lr 0.001 --n_layers 8 --skip=True
python train_graph_inr.py --model INR --dataset_dir dataset/bunny_res2 --n_nodes_in_sample 5000 --lr 0.001 --n_layers 8 --skip=True
python train_graph_inr.py --model Spatial_Graph_INR --dataset_dir dataset/bunny_res2  --emb_dir spatial_embeddings/bunny_res2/...   --emb_name ... --n_nodes_in_sample 5000 --lr 0.001 --n_layers 8 --skip=True

python train_graph_inr.py --dataset_dir dataset/bunny_res2/super_resolution/original --n_fourier 7 --n_nodes_in_sample 5000 --lr 0.001 --n_layers 8 --skip=True
python train_graph_inr.py --model INR --dataset_dir dataset/bunny_res2/super_resolution/original --n_nodes_in_sample 5000 --lr 0.0001 --n_layers 8 --skip=True
python train_graph_inr.py --model INR --dataset_dir dataset/bunny_res2/super_resolution/original --n_nodes_in_sample 5000 --lr 0.0001 --n_layers 8 --skip=True --sine=True --all_sine=True
python train_graph_inr.py --model Spatial_Graph_INR --dataset_dir dataset/bunny_res2/super_resolution/original --emb_dir spatial_embeddings/bunny_res2/...   --emb_name ... --n_nodes_in_sample 5000 --lr 0.001 --n_layers 8 --skip=True
python train_graph_inr.py --model Spatial_Graph_INR --dataset_dir dataset/bunny_res2/super_resolution/original --emb_dir spatial_embeddings/bunny_res2/...   --emb_name ... --n_nodes_in_sample 5000 --lr 0.001 --n_layers 8 --skip=True --input_concat=False

# Conditional INR: reaction-diffusion
python train_graph_inr.py --dataset_dir dataset/bunny_time/ --n_fourier 100 --n_nodes_in_sample 5000 --lr 0.0001 --n_layers 6 --sine=True --all_sine=True --skip=True --time=True

# Conditional INR: multi-protein
# Use the --cut flag to control the size of the dataset
srun python train_graph_inr.py --dataset_dir dataset/proteins --n_fourier 100 --n_nodes_in_sample 5000 --lr 0.0001 --n_layers 6 --latents --latent_dim=8 --sine=True --all_sine=True --skip=True --cut 100

# Weather modelling
# The first 66 eigenvectors are "useless" because the equiangular grid returned by
# GFS is non-uniform on the surface and has higher point density at the poles.
# The effect on the eigenvectors is that the low frequency ones are zero everywhere
# except the poles, and they tend to change a lot when sampling more points.
# To improve stability, we removed the first 66 eigenvectors manually and only
# train on the remaning 34.
# This is basically equivalent to training on the low-frequency eigenvectors
# as we did for the transferability experiments. The setting for the INR is also
# the same (ReLU, 8 layers, higher LR).
python train_graph_inr.py --dataset_dir dataset/weather_time_gustsfc_cut/ --n_fourier 34 --n_nodes_in_sample 5000 --lr 0.001 --n_layers 8 --skip=True --time=True
python train_graph_inr.py --dataset_dir dataset/weather_time_dpt2m_cut/   --n_fourier 34 --n_nodes_in_sample 5000 --lr 0.001 --n_layers 8 --skip=True --time=True
python train_graph_inr.py --dataset_dir dataset/weather_time_tcdcclm_cut/ --n_fourier 34 --n_nodes_in_sample 5000 --lr 0.001 --n_layers 8 --skip=True --time=True
