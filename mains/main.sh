echo "Starting Synthetic"
python main_VCNET_loop.py

echo "Starting IHDP"
python main_ihdp.py

echo "Starting IHDP Continuous"
python main_ihdp_cont_loop.py

echo "Starting TCGA"
python main_tcga.py

echo "END"