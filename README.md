<h2 align="center">NeuSEditor</h2>
<p align="center"><b>From Multi-View Images to Text-Guided Neural Surface Edits</b> <br>
<em>Official Implementation</em></p>

<p align="center">
  <a href="https://neuseditor.github.io/">
    <img src="https://img.shields.io/badge/Project%20Page-Link-blue?style=for-the-badge">
  </a>
</p>

<p align="center">
  <img src="assets/teaser.gif" alt="NeuSEditor Teaser" width="600">
</p>

<p align="center">
  NeuSEditor enables <b>text-guided neural surface editing</b> directly from multi-view images.<br>
  This repository contains the official code and instructions to reproduce our results.
</p>

### Environment Setup

NeuSEditor is tested on NVIDIA RTX A6000, A100, and H100 GPUs.
- **Recommended:** CUDA 11.8 or 12.1, PyTorch 2.1 (see `environment.yml` for details).

To get started, create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate neus_editor
```

You can verify that your GPU is accessible and CUDA versions are correct with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.__version__)"
```


### 📦 Datasets

**NeRF-Synthetic (Blender)**
- **Download:** [Google Drive Link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
- **Usage:** Place the downloaded folders under `load/`, e.g.:  
  ```
  load/nerf_synthetic/lego
  ```

**DTU (Preprocessed by NeuS)**
- **Download:** [Google Drive Link](https://drive.google.com/drive/folders/1Nlzejs4mfPuJYORLbDEUDWlc9IZIbU0C?usp=sharing)
- **Usage:** Use these scans with the DTU config.

---

### 🚀 Quick Start

Below are example commands to launch training for several supported datasets. Adjust command-line flags as needed.

<details>
<summary><strong>NeRF-Synthetic (Blender)</strong></summary>

```bash
python launch.py --config configs/blender.yaml --gpu 0 --train tag=example
```
</details>

<details>
<summary><strong>DTU</strong></summary>

```bash
python launch.py --config configs/dtu.yaml --gpu 0 --train
```
</details>

<details>
<summary><strong>Example DTU <em>(scan24, church prompt)</em></strong></summary>

```bash
python launch.py --config configs/dtu.yaml --gpu 0 \
  tag=scan24_church_gs350 \
  diffusion.guidance_scale=350 \
  diffusion.tgt_prompt="make it a church" \
  dataset.root_dir=../data/DTU/scan24 \
  --train
```
</details>

<details>
<summary><strong>Custom COLMAP</strong></summary>

```bash
python launch.py --config configs/colmap.yaml --gpu 0 --train
```
</details>


### Training on Custom COLMAP Data

To get COLMAP data from custom images, you should have COLMAP installed (see [here](https://colmap.github.io/install.html) for installation instructions). Then put your images in the `images/` folder, and run `scripts/imgs2poses.py` specifying the path containing the `images/` folder. For example:

```bash
python scripts/imgs2poses.py ./load/images # images are in ./load/images
```

This part is adopted from Instant-NSR-PL; please refer to that repository for detailed usage and best practices: https://github.com/bennyguo/instant-nsr-pl

### Acknowledgements

Parts of this codebase and scripts are inspired by or adapted from:

- Instant Neural Surface Reconstruction (Instant-NSR/Instant-NGP Lightning): [https://github.com/bennyguo/instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl)
- PDS: [https://github.com/KAIST-Visual-AI-Group/PDS](https://github.com/KAIST-Visual-AI-Group/PDS)
- threestudio: [https://github.com/threestudio-project/threestudio](https://github.com/threestudio-project/threestudio)

### Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{ibrahimli2026neuseditor,
  title = {NeuSEditor: From Multi-View Images to Text-Guided Neural Surface Edits},
  author = {Ibrahimli, Nail and Kooij, Julian and Nan, Liangliang},
  booktitle = {International Conference on 3D Vision (3DV)},
  year = {2026}
}
```

### License

This project is released under the terms of the license found in `LICENSE`.