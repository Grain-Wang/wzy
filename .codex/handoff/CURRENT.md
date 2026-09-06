# AutoResearch 当前接续状态

更新时间：2026-09-05（Asia/Shanghai）

> **当前有效项目：paper4 / Query-Aware Dynamic Precision for VLMs**
>
> 本文件仅作接续导航。事实优先级：`AGENTS.md` > 当前 Git、论文笔记与实验产物 > 本文件 > `TRANSCRIPT.md`。

## 当前阶段

- 分支：`paper4`。
- 研究状态：**Research Opportunity / CONDITIONAL GO**；尚未达到 Paper Candidate。
- 本轮没有开始模型下载、实现或大规模实验。
- paper4 科学状态单一入口：`paper4/CURRENT.md`。
- 完整文献调研：`paper4/literature/surveys/query_aware_vlm_quantization_literature_review.md`。

## 文献门禁结论

- 宽泛的 `query-aware dynamic mixed-precision quantization` 已发生直接碰撞，不能声称首次提出。
- 最高威胁近邻：QAQ（query-routed weight bit-planes）、PAQ（per-prompt whole-model routing）、TAQ（task-conditioned layer bits）、DP-LLM（runtime layer precision）、PMPD（prefill/decode and prompt-adaptive precision）、MixKVQ（query-aware KV channels）、WindowQuant（prompt-aware visual-window KV precision）。
- VLMQ/TLQ/MBQ/MABA/VLM-PTQ/VVSQ 已覆盖 static Hessian/gradient/token/modality-aware PTQ；QuietPrune/IVTP/LVPruning/SparseVLM 已证明 query-dependent visual relevance；QUOTA/QAPruner/Bi-VLM/ZipVL 已威胁简单 pruning+quantization 组合。
- 尚未找到的交集：固定同一 image，仅改变 query，受控测量跨 vision encoder/projector/LLM attention/MLP 的非 KV 量化敏感度，并证明它改变少量硬件可部署 profile 的最优选择。

## 推荐机会

工作名：**CIQ-PP: Counterfactual Image--Query Precision Profiles for Hardware-Realistic VLM Inference**。

候选方法只在现象成立后推进：从 same-image counterfactual sensitivity residual 构造至多 3--4 个后端原生、预打包 precision profiles；用已有 pooled image/query feature 的轻量交互 router 在 request 边界选 profile；uncertainty rescue 仅作可选安全机制。

## 唯一下一动作

在 Qwen2-VL-2B 上运行 same-image multi-query sensitivity canary：

- 数据：GQA functional question pairs，补充 VQAv2/TextVQA 同图多问；无需新人工标签。
- 干预：W4A16 base，逐 module group restore 到 W8/BF16。
- 指标：teacher-forced NLL/KL、官方答案分数、restore gain、sensitivity-vector distance、measured-cost profile regret；按 image bootstrap。
- 资源：单张 A800 80GB，估计 8--20 GPU-hours、约 2 个自然日。
- 继续阈值：相同实测成本下 per-query oracle >=1.5 个绝对点，或成本下降 >=10%，95% CI 不跨 0，且至少两类 query 复现。
- 停止阈值：oracle <0.5 点且成本差 <5%，或 interaction 被 task ID、答案长度、图像分辨率/prompt template 完全解释。

## 禁止提前扩张

- Canary 1 通过前，不训练复杂 router、不写自定义 kernel、不扩 MoE、不跑完整 benchmark matrix。
- 不能使用“first query-aware quantization”“first task-aware bit allocation”“first prefill/decode asymmetric precision”或简单 token-pruning-plus-quantization 作为 headline novelty。
- 若 canary 失败，应停止 Direction A，而不是调阈值或包装 measurement paper。
