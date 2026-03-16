# ECE567_W2026_Team5_CRL

This repository was created to evaluate the reproducibility of Continual Reinforcement Learning.

---

## Benchmark

* [CORA](https://github.com/AGI-Labs/continual_rl) 

```bash
conda create -n continual_rl python=3.9 -y 
conda activate continual_rl
python -m pip install --upgrade pip setuptools wheel
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/AGI-Labs/continual_rl.git
cd continual_rl
pip install -e .
pip install "gym[accept-rom-license]"
pip install "autorom[accept-rom-license]"
AutoROM
sudo mkdir /scratch
sudo chmod -R 777 /scratch
pip uninstall -y numpy
pip install "numpy==1.23.5"
```

## Baselines

* [EWC](https://arxiv.org/pdf/1612.00796)

* [PC](https://arxiv.org/pdf/1805.06370)

* [CLEAR](https://proceedings.neurips.cc/paper_files/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf) 

## Environments

* Atari (`JB243`)

```bash
conda activate continual_rl

# foreground run
unset LD_LIBRARY_PATH
OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
python main.py --config-file configs/atari/ewc_atari.json --output-dir tmp_atari1
OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
python main.py --config-file configs/atari/pnc_atari.json --output-dir tmp_atari2
OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
python main.py --config-file configs/atari/clear_atari.json --output-dir tmp_atari3

# alternatively, background run
mkdir -p tmp_atari1
nohup bash -lc 'unset LD_LIBRARY_PATH; OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python main.py --config-file configs/atari/ewc_atari.json --output-dir tmp_atari1' > nohup_ewc_atari1.out 2>&1 &
mkdir -p tmp_atari2
nohup bash -lc 'unset LD_LIBRARY_PATH; OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python main.py --config-file configs/atari/pnc_atari.json --output-dir tmp_atari2' > nohup_pnc_atari2.out 2>&1 &
mkdir -p tmp_atari3
nohup bash -lc 'unset LD_LIBRARY_PATH; OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python main.py --config-file configs/atari/clear_atari.json --output-dir tmp_atari3' > nohup_clear_atari3.out 2>&1 &

```

* Procgen

* Minihack

* CHORES

* [Nethack](https://github.com/NetHack-LE/nle?tab=readme-ov-file) 

## License

* MIT License
