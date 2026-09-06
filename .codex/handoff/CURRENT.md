# AutoResearch 当前接续状态

更新时间：2026-09-06（Asia/Shanghai）

> **当前有效项目：paper4 / CIQ-PP C01 Canary Validation**
>
> 本文件仅作接续导航。事实优先级：`AGENTS.md` > 当前 Git、论文笔记与实验产物 > 本文件 > `TRANSCRIPT.md`。

## 当前阶段

- 分支：`paper4`。
- 研究状态：**Research Opportunity / CONDITIONAL GO**；尚未达到 Paper Candidate。
- C01-S0 integrity smoke test：**PASS**。C01-S1 未运行，科学假设未被检验。
- 上一阶段方向决策已提交为 `09a613e`；远程 push 被当前工具的远端信任策略阻止，本地提交未丢失。
- paper4 科学状态单一入口：`paper4/CURRENT.md`。
- 完整文献调研：`paper4/literature/surveys/query_aware_vlm_quantization_literature_review.md`。

## C01-S0 事实

- 模型：`Qwen/Qwen2-VL-2B-Instruct`，revision
  `895c3a49bc3fa70a340399125c650a463535e71c`；两分片 SHA-256 已在本地和远端核对。
- 环境：Python 3.12.13、PyTorch 2.5.0+cu121、Transformers 4.55.2、CUDA 12.1、单张 A800 80GB。
- 数据：GQA balanced test-dev，8 images / 32 questions；四类 query 均覆盖。
- 冻结 sample manifest SHA-256：
  `cfea83045278c17d8f1428b381ef1bee181fc69bf3a128ef41165b38ec3bdc05`。
- module manifest：13 groups / 326 tensors；SHA-256
  `db080e473fb6f9fa4204b1372ace0a5793644a71f41566a99b1847f696e705f6`。
- integrity checks 全部通过：BF16 repeatability、empty no-op、single-group isolation、exact restore、all-W4、evaluator、answer-only NLL、same-image grouping、vision-cache equivalence、freeze/leakage。
- 成功运行 0.0681 A800 GPU-hour；含两次在产生 sensitivity output 前停止的实现门禁尝试，累计保守估计低于 0.10 GPU-hour。S0 进程已释放。
- 详细报告：`paper4/experiments/02_canary/C01/S0_REPORT.md`；小型机器摘要：`paper4/results/processed/C01/S0/integrity_summary.json`。
- 模型、数据、环境、cache 与 raw 输出保留在仓库内忽略目录，不得加入 Git。
- 修复：真实 Transformers decoder wrapper 路径、cuBLAS 确定性配置硬门禁、GQA nested type metadata 优先级。
- 剩余关注：S0 小切片 BF16 normalized exact 为 4/32；这不是 S0 gate，但 S1 必须先执行预注册 baseline-validity early stop。

## 文献门禁结论

- 宽泛的 `query-aware dynamic mixed-precision quantization` 已发生直接碰撞，不能声称首次提出。
- 最高威胁近邻：QAQ（query-routed weight bit-planes）、PAQ（per-prompt whole-model routing）、TAQ（task-conditioned layer bits）、DP-LLM（runtime layer precision）、PMPD（prefill/decode and prompt-adaptive precision）、MixKVQ（query-aware KV channels）、WindowQuant（prompt-aware visual-window KV precision）。
- VLMQ/TLQ/MBQ/MABA/VLM-PTQ/VVSQ 已覆盖 static Hessian/gradient/token/modality-aware PTQ；QuietPrune/IVTP/LVPruning/SparseVLM 已证明 query-dependent visual relevance；QUOTA/QAPruner/Bi-VLM/ZipVL 已威胁简单 pruning+quantization 组合。
- 尚未找到的交集：固定同一 image，仅改变 query，受控测量跨 vision encoder/projector/LLM attention/MLP 的非 KV 量化敏感度，并证明它改变少量硬件可部署 profile 的最优选择。

## 推荐机会

工作名：**CIQ-PP: Counterfactual Image--Query Non-KV Precision Profiles for Hardware-Realistic VLM Inference**。

候选方法只在现象成立后推进：从 same-image counterfactual sensitivity residual 构造至多 3--4 个后端原生、预打包 precision profiles；用已有 pooled image/query feature 的轻量交互 router 在 request 边界选 profile；uncertainty rescue 仅作可选安全机制。

## 唯一下一动作

等待用户明确授权 C01-S1。授权后严格按
`paper4/experiments/02_canary/C01/README.md` 的已锁定协议执行 coarse confirmatory probe；先检查 BF16 baseline validity 与 W4 proxy dynamic range，再运行完整 S1 条件。不得自行改写 positive / negative / gray-zone 门槛。

## 禁止提前扩张

- C01-S1 通过前，不训练复杂 router、不写自定义 kernel、不扩 MoE、不跑完整 benchmark matrix。
- 不能使用“first query-aware quantization”“first task-aware bit allocation”“first prefill/decode asymmetric precision”或简单 token-pruning-plus-quantization 作为 headline novelty。
- 若 canary 失败，应停止 Direction A，而不是调阈值或包装 measurement paper。
