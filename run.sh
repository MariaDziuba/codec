#pip install -r requirements.txt

# wandb init
#python EntropySetup.py build_ext --inplace
#pip install .

# Run train
#python -m src.scripts.train --config_file ./configs/train/gelu_ae.yml

# Run inference
python -B -m src.scripts.inference --config_file ./configs/inference/gelu_ae.yaml