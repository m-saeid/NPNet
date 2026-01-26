```markdown
# NPNet: A Non-Parametric Network with Adaptive Gaussian–Fourier Positional Encoding

Official implementation of the paper:  
**“NPNet: A Non-Parametric Network with Adaptive Gaussian–Fourier Positional Encoding for 3D Classification and Segmentation”**  

---

## 🧠 Overview
NPNet is a fully non-parametric network for 3D point cloud analysis.  
It introduces an **adaptive Gaussian–Fourier positional encoding** where kernel width and blending weights are dynamically set from input geometry, ensuring robustness across varying scales and densities.  

- 🚫 **No trainable parameters**  
- ⚡ **Competitive classification and segmentation performance**  
- 💾 **Low memory footprint and fast inference**  
- 📊 **State-of-the-art among non-parametric methods** on ModelNet40, ModelNet-R, ScanObjectNN, ShapeNetPart, and few-shot ModelNet40.  

---

## 📁 Repository Structure

```
.
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
│
├── models
├── run.sh
└── run\_comparison.sh
```

---

## 📦 Supported Datasets
- ModelNet40  
- ModelNet-R  
- ScanObjectNN  
- ShapeNetPart  
- ModelNet40 Few-Shot (5-way, 10-way, 10/20-shots)  

---

## 🛠️ Installation
```bash
# Clone the repo
git clone https://github.com/anonymous/NPNet.git
cd NPNet

# Install dependencies
pip install torch torchvision
pip install -r requirements.txt

# Compile PointNet++ ops
cd pointnet2_ops_lib
pip install .
cd ..
````

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

## 🙌 Acknowledgements

* [PointNet++](https://arxiv.org/abs/1706.02413)
* [Point-NN](https://arxiv.org/abs/2303.08134)
* [Point-GN](https://arxiv.org/abs/2003.01251)



