# AutoResearch 当前接续状态

更新时间：2026-09-06（Asia/Shanghai）

> **当前有效项目：paper4 / CIQ-PP C01 Canary Validation**
>
> 本文件仅作接续导航。事实优先级：`AGENTS.md` > 当前 Git、论文笔记与实验产物 > 本文件 > `TRANSCRIPT.md`。

## 当前阶段

- 分支：`paper4`。
- 研究状态：**Research Opportunity / C01-S1 INCONCLUSIVE**；尚未达到 Paper Candidate。
- C01-S0 integrity smoke test：**PASS**。C01-S1 coarse confirmatory probe：**INCONCLUSIVE**；C01-S2 未获授权、未运行。
- 方向决策与 S0 已有本地提交；S1 完成后按当前 Git 状态核验。本仓库规则禁止向外部仓库 push，不能把未同步误记为实验失败。
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

## C01-S1 事实

- S1-A baseline-validity：**PASS**；S1-B W4 proxy dynamic-range：**PASS**，未使用 proxy correction。
- 模型与 revision 延续 S0。正式 GQA manifest：90 images / 540 questions / 270 repeated image×family cells；SHA-256
  `6a7d8237438bd1764cb8a9341fdafad0e5d43dbbba448962bdd0f532993a63b0`。
- BF16 official 60.00%；all-W4 official 59.26%；all-W4 mean ΔNLL 0.032510，mean gold-position JS 0.012857，answer flip 14.07%。
- 13-group primary controlled interaction partial R² = -0.603177，image-bootstrap 95% CI [-0.671533, -0.523433]，within-image permutation p=0.051，standardized effect size=1.5682。interaction gate **FAIL**。
- +20% exact-byte budget 下，executed per-query oracle 比最强 global/task/image control 高 4.17 official-score points（95% CI [+0.83,+8.33]）与 264.57 pp relative NLL recovery（95% CI [+101.81,+2442.28]）。profile gate PASS，但 NLL recovery 的小分母使幅度和 CI 很不稳定。
- same-family / cross-family cosine distance 分别为 0.90654 / 0.93513；bootstrap top-3 median Jaccard 0.500，lower-95% 0.200，exact identity 38.85%。
- 结果为 **INCONCLUSIVE**：不是 STRONG PASS；也因 oracle headroom 较大而不符合 strict robust NEGATIVE。不能将 oracle headroom 表述为已验证的稳定 image×query interaction。
- recoverable BF16-correct/W4-wrong tail 为 26/540，覆盖 fine-grained、reasoning、spatial 三类；仅记录 backup 可测试，本轮未运行 backup。
- 成功 GPU interval 共 1.8090 A800 GPU-hours，peak process allocation 9,074,570,752 bytes；0 次 GPU inference retry。两次 post-processing bug 与一次 random-priority/additive-estimate 字段误标均在复用原始推理输出后修复并测试，未改变 profile choices 或 gate statistics。
- 报告：`paper4/experiments/02_canary/C01/S1_REPORT.md`；机器摘要：`paper4/results/processed/C01/S1/s1_confirmatory_summary.json`；四张图位于 `paper4/results/figures/C01/`。

## 文献门禁结论

- 宽泛的 `query-aware dynamic mixed-precision quantization` 已发生直接碰撞，不能声称首次提出。
- 最高威胁近邻：QAQ（query-routed weight bit-planes）、PAQ（per-prompt whole-model routing）、TAQ（task-conditioned layer bits）、DP-LLM（runtime layer precision）、PMPD（prefill/decode and prompt-adaptive precision）、MixKVQ（query-aware KV channels）、WindowQuant（prompt-aware visual-window KV precision）。
- VLMQ/TLQ/MBQ/MABA/VLM-PTQ/VVSQ 已覆盖 static Hessian/gradient/token/modality-aware PTQ；QuietPrune/IVTP/LVPruning/SparseVLM 已证明 query-dependent visual relevance；QUOTA/QAPruner/Bi-VLM/ZipVL 已威胁简单 pruning+quantization 组合。
- 尚未找到的交集：固定同一 image，仅改变 query，受控测量跨 vision encoder/projector/LLM attention/MLP 的非 KV 量化敏感度，并证明它改变少量硬件可部署 profile 的最优选择。

## 推荐机会

工作名：**CIQ-PP: Counterfactual Image--Query Non-KV Precision Profiles for Hardware-Realistic VLM Inference**。

候选方法只在现象成立后推进：从 same-image counterfactual sensitivity residual 构造至多 3--4 个后端原生、预打包 precision profiles；用已有 pooled image/query feature 的轻量交互 router 在 request 边界选 profile；uncertainty rescue 仅作可选安全机制。

## 唯一下一动作

不得执行 C01-S2。按 gray-zone 规则，只能在新一轮明确授权后做一次 bounded、preregistered sample expansion，并继续使用已经通过的 W4 proxy；不能同时改 proxy。如果扩展仍不通过 interaction gate，应 downgrade CIQ-PP。不得训练 router 或启动 backup experiment。

## 禁止提前扩张

- C01-S1 未 STRONG PASS，不训练复杂 router、不写自定义 kernel、不扩 MoE、不跑完整 benchmark matrix，也不执行 S2。
- 不能使用“first query-aware quantization”“first task-aware bit allocation”“first prefill/decode asymmetric precision”或简单 token-pruning-plus-quantization 作为 headline novelty。
- 若 canary 失败，应停止 Direction A，而不是调阈值或包装 measurement paper。
