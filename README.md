# medmnist-mini-bench — Minimal, Reproducible Baselines

> 一键跑通 MedMNIST 的 2D/3D 小型基线，用最小工作量呈现“能跑、能看、能复现”。

[![CI](https://img.shields.io/github/actions/workflow/status/xlwang208/medmnist-mini-bench/ci.yml?branch=main)](https://github.com/xlwang208/medmnist-mini-bench/actions)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

## 🎯 任务目的（为什么存在这个仓库）
- **能跑**：一条命令拉数据、训练 1 个 epoch、输出指标与图表。
- **能看**：README 给清晰的命令、目录、结果产物（metrics.json、混淆矩阵）。
- **能复现**：固定随机种子、版本依赖、最小 CI 绿勾；配置化（YAML）；结果落盘。

面向场景：申请博士/岗位时展示**工程与实验素养**：数据加载 → 训练 → 评估 → 产出 → CI。

## 🔧 安装
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .[dev]  # 或：pip install -r requirements.txt
```

> 若在中国大陆网络环境，您可优先配置 PyTorch 的国内镜像源后再安装 torch/torchvision。

## ▶️ 一键 Demo（建议从 PathMNIST 开始）
```bash
# 2D：PathMNIST（9 类，彩色 28x28）
python scripts/run_demo.py --dataset pathmnist --epochs 1

# 3D：OrganMNIST3D（11 类，体素 28x28x28）
python scripts/run_demo.py --dataset organmnist3d --model cnn3d --epochs 1
```

输出保存在 `outputs/<dataset>/<model>/<timestamp>/`：
- `metrics.json`：acc/auc 等
- `confusion_matrix.png`：混淆矩阵
- `best.pt`：最佳权重（依据验证集 acc）

> 机器较慢或用于 CI 时，可加 `--limit-samples 512` 加速；或把 `--epochs` 设为 1。

## 📦 结构
```
medmnist-mini-bench/
├── src/bench/...
├── scripts/run_demo.py
├── configs/{path,organ3d}.yaml
├── tests/test_sanity.py
├── pyproject.toml / requirements.txt
├── .github/workflows/ci.yml
└── README.md / LICENSE / CITATION.cff
```

## ⚙️ 可复现实验（配置化）
```bash
# 使用配置文件（等价于直接传参）
python scripts/run_demo.py --config configs/path.yaml
python scripts/run_demo.py --config configs/organ3d.yaml
```

## 🧪 CI
仓库已包含 GitHub Actions（安装依赖 → 运行 pytest → 以小样本跑一遍 demo）。
- 将 README 顶部徽章里的 `yourname/medmnist-mini-bench` 替换为你的 GitHub 路径。
- 如果 CI 超时，可在 `.github/workflows/ci.yml` 中把 `--limit-samples` 再减小。

## 🔍 结果说明
- 本仓库**不承诺**达到论文级 SOTA，目标是最小、透明、可复现的工程基线。
- AUC/Acc 依赖随机种子与训练时长，建议自行在 README 中记录你的复现实验结果。

## 📄 引用
若本仓库对你有帮助：请在你的代码或文档中添加链接，或使用 `CITATION.cff`。

---

### 致谢
数据集来自 MedMNIST 项目；模型实现参考 PyTorch 官方样例思路（本仓库手写极简网络以便教学与复现）。
