# AutoResearch

AutoResearch 是最外层主仓库、Git 仓库和自主科研工作区，由仓库级 [`AGENTS.md`](AGENTS.md) 统一约束。目标是在有限资源内形成以算法创新为核心、具备强 CCF-C / 弱 CCF-B 竞争力的完整论文证据链。

[`tools/`](tools/) 是通用研究执行工具箱，不拥有独立研究目标；每篇论文的研究问题、文献、实验记录、结果、评审与论文材料保存在对应的 `paper*/` 目录中。

## 当前分支：paper4

`paper4` 当前研究 Query-Aware Dynamic Precision for VLMs。文献与碰撞审计已完成，宽泛的 query-/task-/runtime-aware quantization 已有强近邻；当前保留的收缩方向是 **CIQ-PP: Counterfactual Image--Query Precision Profiles for Hardware-Realistic VLM Inference**。

当前状态：

- 研究阶段：Research Opportunity
- 当前决策：CONDITIONAL GO
- Paper Candidate Gate：FAIL / UNVERIFIED
- 当前唯一动作：先验证同一图像更换 query 是否显著改变非 KV 模块的量化敏感度和最优硬件 profile
- 权威状态入口：[`paper4/CURRENT.md`](paper4/CURRENT.md)

## paper4 研究工作区

```text
paper4/
├── CURRENT.md
├── literature/
│   ├── papers/
│   ├── notes/
│   └── surveys/
├── ideas/
│   ├── candidates/
│   └── archived/
├── analysis/
│   ├── literature_gap.md
│   └── novelty_matrix.md
├── experiments/
│   ├── 01_baseline/
│   ├── 02_canary/
│   ├── 03_method/
│   └── 04_ablation/
├── configs/
├── src/
├── tests/
├── results/
│   ├── raw/
│   ├── processed/
│   ├── tables/
│   └── figures/
└── paper/
    ├── outline.md
    ├── related_work.md
    ├── method.md
    └── experiments.md
```

## 路径用途

| 路径 | 用途 |
| --- | --- |
| `AGENTS.md` | 整个仓库的权威研究与执行规则 |
| `.codex/handoff/` | 脱敏的 Codex 接续资料；使用前必须与当前分支源码和 Git 状态核对 |
| `tools/` | 通用研究执行工具箱 |
| `paper4/CURRENT.md` | paper4 当前科学状态、门禁、canary 和下一动作的单一入口 |
| `paper4/literature/papers/` | 文献原文与经许可保存的附件 |
| `paper4/literature/notes/` | 单篇论文的结构化阅读笔记 |
| `paper4/literature/surveys/` | 跨论文综述、技术谱系与系统性文献审计 |
| `paper4/ideas/candidates/` | 尚在验证的候选研究方向与机制假设 |
| `paper4/ideas/archived/` | 已否定、放弃或被碰撞淘汰的方向及原因 |
| `paper4/analysis/` | literature gap、novelty matrix、research questions 和阶段科学判断 |
| `paper4/experiments/01_baseline/` | baseline 复现、缺陷验证与公平性检查 |
| `paper4/experiments/02_canary/` | 最小可证伪 canary 实验协议与执行入口 |
| `paper4/experiments/03_method/` | 通过 canary 后的候选方法实验 |
| `paper4/experiments/04_ablation/` | 消融、边界、鲁棒性和机制诊断实验 |
| `paper4/configs/` | 可复现实验配置；不得存放认证信息 |
| `paper4/src/` | paper4 可复用算法与实验实现代码 |
| `paper4/tests/` | 自动化测试 |
| `paper4/results/raw/` | 运行程序直接产生、未经变换的原始输出 |
| `paper4/results/processed/` | 可由 raw 结果重建的清洗、聚合或统计数据 |
| `paper4/results/tables/` | 论文与分析使用的最终表格 |
| `paper4/results/figures/` | 论文与分析使用的最终图 |
| `paper4/paper/` | 长期维护的论文 outline、related work、method 和 experiments 草稿 |

实验代码与结果必须分离；原始结果与处理后结果不得混放。空目录通过 `.gitkeep` 纳入版本控制，在目录产生正式文件后可删除相应占位文件。不再使用语义模糊的 `paper4/steps/`；新文件应直接进入最匹配的语义目录。

## 跨机器继续 paper4

新克隆：

```bash
git clone -b paper4 https://github.com/Grain-Wang/wzy.git AutoResearch
cd AutoResearch
codex -C .
```

已有仓库：

```bash
git fetch origin
git switch paper4
git pull --ff-only
codex -C .
```

新会话应先读取 `AGENTS.md`，再核对当前分支、源码、测试、结果与 Git 状态。历史 handoff 资料只有在确认属于 paper4 后才能作为当前研究状态使用。

## 环境与依赖

项目使用 Python 3.12。通用工具依赖由 `tools/pyproject.toml` 管理；paper4 开始实现实验后，应在 `paper4/pyproject.toml` 中声明并锁定实际运行依赖，不得只在某台机器临时安装。

首选 Conda 环境名：

```bash
conda create -n auto_research python=3.12 -y
conda activate auto_research
```

安装通用工具：

```bash
cd tools
python -m pip install -e ".[dev]"
```

## 研究与提交约束

- 核心贡献必须是可明确描述和验证的新算法、目标函数、决策机制或优化过程，不能把工程修复、配置问题或单纯 benchmark 包装成论文贡献。
- paper4 不继承其他论文项目的数据集、研究门禁、reviewer 状态或 handoff 结论。
- 优先验证 baseline 缺陷、最小算法对象和强公平 baseline；未通过 Paper Candidate Gate 前不启动大规模实验。
- 新增代码必须可由命令行重建实验，并记录配置、版本、随机种子和机器信息。
- 凭据、SSH 信息、私钥、本地绝对路径、环境目录、缓存和未授权大型数据不得提交。

## 代码检查

新增或修改研究代码后，按 `AGENTS.md` 与 paper4 的项目配置运行至少：

```bash
ruff check .
black --check .
pytest tests/
```

如果仓库级检查被历史无关代码阻断，应同时报告仓库级失败与 paper4 范围结果，不得把范围内通过冒充全仓库通过。

若本 README 与 `AGENTS.md` 冲突，以 `AGENTS.md` 为准。
