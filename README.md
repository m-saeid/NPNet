# 🚀 NPNet: A Non-Parametric Network with Adaptive Gaussian–Fourier Positional Encoding for 3D Classification and Segmentation

**Official implementation** of the IEEE IV 2026 paper:  
**“NPNet: A Non-Parametric Network with Adaptive Gaussian–Fourier Positional Encoding for 3D Classification and Segmentation”**  
by Mohammad Saeid, Amir Salarpour, Pedram MohajerAnsari, Mert D. Pesé.


<p align="center">
  <a href="https://arxiv.org/abs/2602.00542">
    <img src="https://img.shields.io/badge/Paper-arXiv-brightgreen" alt="Paper"/> <!-- on arXiv"/> -->
  </a>
  <a href="https://m-saeid.github.io/NPNet">
    <img src="https://img.shields.io/badge/Project-Homepage-red" alt="Project Homepage"/>
  </a>
</p>
  <!--
  <a href="https://www.youtube.com/watch?v=7ziipjpdth0&list=PLvWl5fdJgzQxaF0v4egv1cdrstl8N7fEM&index=2">
    <img src="https://img.shields.io/badge/Video-Presentation-blue" alt="YouTube Presentation"/>
  </a>
    <a href="https://github.com/m-saeid/ModeNetR_PointSkipNet/blob/main/images/ModelNet%E2%80%91R%20%26%20Point%E2%80%91SkipNet.pdf" target="_blank">
      <img src="https://img.shields.io/badge/Presentation-PDF-orange" alt="Presentation PDF"/>
    </a>
  -->
  <!--
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange" alt="PyTorch Framework"/>
  </a>
  <a href="https://github.com/m-saeid/ModeNetR_PointSkipNet/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0 License"/>
  </a>
  -->

---

NPNet is a fully non-parametric network for 3D point cloud analysis.  
It introduces an **adaptive Gaussian–Fourier positional encoding** where kernel width and blending weights are dynamically set from input geometry, ensuring robustness across varying scales and densities.  

- 🏆 **State-of-the-art non-parametric method** on ModelNet40, ModelNet-R, ScanObjectNN, ShapeNetPart, and few-shot ModelNet40  
- 🚫 **No trainable parameters**  
- ⚡ **Competitive classification and segmentation performance** 
- 💾 **Low memory footprint and fast inference**  

---

## 📁 Repository Structure

```
├── comparison
│   ├── data
│   ├── NPNet
│   ├── Point\_GN
│   ├── Pointnet\_Pointnet2
│   ├── Point\_NN
│   ├── scripts
│   └── tasks
│
├── data
├── datasets
├── models
├── ...
├── run.sh
├── train_np_cls_mn.py
├── train_np_cls_scan.py
├── train_np_seg.py
└── ...
```

---

## 📦 Supported Datasets
- ModelNet40  
- ModelNet-R  
- ScanObjectNN  
- ShapeNetPart  
- ModelNet40 Few-Shot (5-way, 10-way, 10/20-shots)  

---

## 📦 Dataset Preparation

**Download Datasets** and place them under `dataset/` with the following folder structure:
   ```bash
    ├── datasets
    │   ├── h5\_files
    │   │   ├── main\_split
    │   │   ├── main\_split\_nobg
    │   ├── modelnet40\_ply\_hdf5\_2048
    │   ├── modelnet\_fewshot
    │   │   ├── 10way\_10shot
    │   │   ├── 10way\_20shot
    │   │   ├── 5way\_10shot
    │   │   └── 5way\_20shot
    │   ├── modelnetR\_ply\_hdf5\_2048
    │   └── shapenetcore\_partanno\_segmentation\_benchmark\_v0\_normal
    │       ├── train\_test\_split
    │       └── util
   ```

---

## 🛠️ Installation

1. **Clone this repository**
    ```bash
    git clone https://github.com/m-saeid/NPNet.git
    cd NPNet
    ```
    
2. **Install Python dependencies**
   ```bash
   pip install torch torchvision
   pip install -r requirements.txt
   ```
   
3. **Install `gcc-10 g++-10`**
   ```bash
   sudo apt update
   sudo apt install gcc-10 g++-10
   
   export CC=gcc-10
   export CXX=g++-10
   ```

4. **Install `pointnet2_ops_lib`**

   ```bash
   cd pointnet2_ops_lib
   pip install .
   cd ..
   ```

5. **Verify CUDA & GPU setup**

   * Ensure CUDA 11.x or 12.x is installed and matches your GPU drivers.
   * Confirm with:

     ```bash
     nvidia-smi
     nvcc --version
     ```

---

## 🎯 Usage

### Run all experiments

```bash
bash run.sh
```

### Classification - ModelNet40

```bash
python train_np_cls_mn.py --dataset modelnet40       # acc: 85.45
python train_np_cls_mn.py --dataset modelnet-R       # acc: 85.65
```

### Few-Shot Classification - ModelNet40

```bash
python train_np_cls_mn.py --dataset modelnet40fewshot --n_way 5 --k_shots 10   # acc: 92.0
python train_np_cls_mn.py --dataset modelnet40fewshot --n_way 5 --k_shots 20   # acc: 93.2
python train_np_cls_mn.py --dataset modelnet40fewshot --n_way 10 --k_shots 10  # acc: 82.5
python train_np_cls_mn.py --dataset modelnet40fewshot --n_way 10 --k_shots 20  # acc: 87.6
```

### Classification - ScanObjectNN

```bash
python train_np_cls_scan.py --split OBJ_BG       # acc: 86.1
python train_np_cls_scan.py --split OBJ_ONLY     # acc: 86.1
python train_np_cls_scan.py --split PB_T50_RS    # acc: 84.9
```

### Segmentation - ShapeNet

```bash
python train_np_seg.py --dataset shapenetpart    # acc: 73.5
```

---
## 📊 Results

### 🧠 Classification

| Dataset | Accuracy | Parameters |
|--------|---------:|-----------:|
| ModelNet40 | 85.45% | 0 |
| ModelNet-R | 85.65% | 0 |
| ScanObjectNN OBJ_BG | 86.1% | 0 |
| ScanObjectNN OBJ_ONLY | 86.1% | 0 |
| ScanObjectNN PB_T50_RS | 84.9% | 0 |

### 🎯 Few-Shot Classification

| Dataset | Accuracy | Parameters |
|--------|---------:|-----------:|
| ModelNet40 5-way 10-shot | 92.0% | 0 |
| ModelNet40 5-way 20-shot | 93.2% | 0 |
| ModelNet40 10-way 10-shot | 82.5% | 0 |
| ModelNet40 10-way 20-shot | 87.6% | 0 |

### ✂️ Segmentation (ShapeNet)

| Dataset | mIoU | Parameters |
|--------|-----:|-----------:|
| ShapeNetPart | 73.5% | 0 |


---

## ⚡ Efficiency Comparison

NPNet achieves lower FLOPs, reduced GPU memory usage, and faster inference compared with other non-parametric baselines.  

To reproduce efficiency profiling:  
```bash
cd comparison
bash scripts/run.sh
```


## 📊 Efficiency Comparison (RTX 3090)

| Model    | Dataset  | GFLOPs | GPU Mem. (MB) | Params (M) | Inference (ms) | Points |
| -------- | -------- | ------ | ------------- | ---------- | -------------- | ------ |
| NPNet    | ModelNet | 0.0021 | 99.1          | 0          | 3.86           | 1024   |
| Point-NN | ModelNet | 0.0027 | 161.0         | 0          | 4.44           | 1024   |
| Point-GN | ModelNet | 0.0021 | 161.0         | 0          | 5.80           | 1024   |
| NPNet    | ShapeNet | 0.0045 | 256.4         | 0          | 5.63           | 1024   |
| Point-NN | ShapeNet | 0.0054 | 442.9         | 0          | 16.83          | 1024   |

---

## 📊 Ablation Study

To reproduce Ablation Study:  
```bash
cd comparison
bash scripts/run_ablation.sh
```

---



## 📝 Citation

```bibtex
@article{saeid2026npnet,
  title={NPNet: A Non-Parametric Network with Adaptive Gaussian-Fourier Positional Encoding for 3D Classification and Segmentation},
  author={Saeid, Mohammad and Salarpour, Amir and MohajerAnsari, Pedram and Pes{\'e}, Mert D},
  journal={arXiv preprint arXiv:2602.00542},
  year={2026}
}
```

---

## 📬 Contact

📧 Questions? Reach out to: **[imm.saeid@gmail.com](imm.saeid@gmail.com)**

---

## 🙌 Acknowledgements

* [PointNet++](https://arxiv.org/abs/1706.02413)
* [Point-NN](https://arxiv.org/abs/2303.08134)
* [Point-GN](https://arxiv.org/abs/2003.01251)














