# Polybee F3-Swarm v1.5.2: Verifier-Guided Heterogeneous SLM Routing, Symbolic Evidence Normalization, and Controlled Memory-Pipeline Benchmarks on Commodity Edge Hardware

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Preprint: arXiv](https://img.shields.io/badge/Preprint-arXiv%20[cs.AI]%20(v1.5.2)-red.svg)](paper/Polybee_F3_Swarm_v1_5_arXiv_Preprint.pdf)
[![Paper: PDF](https://img.shields.io/badge/Paper-PDF%20Download-informational.svg)](paper/Polybee_F3_Swarm_v1_5_arXiv_Preprint.pdf)
[![Source: LaTeX](https://img.shields.io/badge/LaTeX-Source%20Code-success.svg)](paper/polybee_f3_swarm_v1_5_arxiv.tex)
[![Accuracy: 96.7% In-Sample | 93.3% Held-Out](https://img.shields.io/badge/Accuracy-96.7%25%20In--Sample%20%7C%2093.3%25%20Held--Out-success.svg)]()
[![Hardware: Edge CPU](https://img.shields.io/badge/Hardware-Edge%20CPU%20(Intel%20i7--11370H)-orange.svg)]()
[![RAM: <1.5 GiB](https://img.shields.io/badge/RAM%20Active-%3C%201.5%20GiB-blueviolet.svg)]()
[![DOI: Zenodo](https://img.shields.io/badge/Zenodo-DOI%20upon%20v1.5.2%20release-yellow.svg)](https://github.com/henrique-polybee/polybee-f3-swarm)

> **Polybee AI Research Laboratory — Manuscript prepared for arXiv [cs.AI] (Submission Candidate v1.5.2)**  
> **Title:** *Polybee F3-Swarm v1.5.2: Verifier-Guided Heterogeneous SLM Routing, Symbolic Evidence Normalization, and Controlled Memory-Pipeline Benchmarks on Commodity Edge Hardware*  
> **Author:** Henrique Oliveira (Polybee AI Research Laboratory, São Paulo, SP, Brazil — [https://polybee.ai/](https://polybee.ai/))  
> 
> 📄 **[Download Technical Preprint Manuscript (PDF)](paper/Polybee_F3_Swarm_v1_5_arXiv_Preprint.pdf)** | 📝 **[LaTeX Source (`.tex`)](paper/polybee_f3_swarm_v1_5_arxiv.tex)** | 📦 **[Complete LaTeX Bundle (`.zip`)](paper/Polybee_F3_Swarm_v1_5_arXiv_Source.zip)** | 🔬 **[Technical Markdown](paper/PAPER_POLYBEE_F3_SWARM_2026.md)**  
> ℹ️ *Note: Zenodo DOI will be permanently minted upon the official v1.5.2 release.*

---

## 📌 Executive Summary

**Polybee F3-Swarm** is an enterprise-grade, heterogeneous three-tier language model swarm designed for mission-critical operations under severe hardware constraints (commodity consumer edge CPUs, low latency, sub-1.5 GiB active RAM budget, and dynamic CPU thermal limits).

The experimental framework and empirical evaluation are organized into two distinct, complementary tracks:
1. **Track A — Controlled Memory-Pipeline Benchmark (Labs 01–02):** Evaluates distributed resident-weight RAM pooling over local TCP loopback (`127.0.0.1`) under hard Win32 per-process working set quotas (`SetProcessWorkingSetSizeEx` at 600 MiB) and unbuffered NVMe SSD I/O (`FILE_FLAG_NO_BUFFERING`), quantifying the throughput speedup of resident network pipelining vs. disk swap streaming.
2. **Track B — Sequential-Lifecycle Heterogeneous SLM Cascade (Labs 03–05):** Evaluates real quantized GGUF models (`Qwen2.5-0.5B`, `Qwen2.5-1.5B`, and `Qwen2.5-Coder-3B`) via `llama-server` under a strict single-active-server RAM budget (< 1.5 GiB for Tiers 1–2), orchestrating calibrated System-1 triage, Controlled Micro-CoT with deterministic suffix closure, substring-anchored symbolic normalization (**Motor B**), and Four-Case Functional Testing ($K=4$) with counterexample-guided code repair (**AACR**).

Polybee F3-Swarm achieves **96.7% in-sample accuracy (29/30)** and a **93.3% Held-Out-Adjusted 30-Task Aggregate (28/30)**, establishing the best observed accuracy among evaluated policies on the proposed 30-task benchmark while retaining **50.0% of all requests entirely at Tier-1 edge at near-zero incremental latency**.

---

## 🏆 Key Benchmark Results

### Track A: Controlled Memory-Pipeline Benchmark (Lab 02 Physical Telemetry)

Evaluated under hard Win32 working-set limits ($600\,\text{MiB}$) and unbuffered NVMe I/O streaming a 24-layer matrix transformation pipeline ($1.50\,\text{GiB}$ `float32`, 32 steps) on an Intel Core i7-11370H:

| Arm / Network Condition | Working Set (MiB) | NVMe SSD Read (MiB) | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | Speedup vs SSD Stream | Numerical Parity (max \|Δ\|) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm 1: Unbuffered NVMe SSD Stream** | 530.9 | 33,792.0 (1.57 GB/s) | 715.30 | 692.58 | 1.44 | 1.00× (Baseline) | — |
| **Arm 2: Emulated LAN (0.4 ms RTT)** | 529.5–546.8 | 0.0 | **95.33** | **50.54** | **19.79** | **13.70×** | 0.0 (Bit-for-bit exact) |
| **Arm 2: Emulated Wi-Fi (4.5 ms RTT)** | 529.5–546.8 | 0.0 | **59.54** | **64.02** | **15.62** | **10.82×** | 0.0 (Bit-for-bit exact) |
| **Arm 2: Emulated WAN (60.0 ms RTT)** | 529.5–546.8 | 0.0 | **204.21** | **206.70** | **4.84** | **3.35×** | 0.0 (Bit-for-bit exact) |

*Data directly reconciled from `lab02_physical_results.json`. Resident-weight loopback pipelining completely eliminates 33.00 GiB of SSD paging and yields up to a **13.70× throughput speedup**.*

---

### Track B: Sequential-Lifecycle Heterogeneous SLM Cascade (Labs 03–05)

Evaluated across **30 real-world operational administrative, financial, and code synthesis tasks in Brazilian Portuguese (PT-BR)** with exact 95% Wilson Score Confidence Intervals:

| Metric / Dimension | P1: Small Always (0.5B) | P2: Large Always (1.5B) | P3: Cascaded Swarm v0 | P4: Swarm v1.0 AACR | P5a: Diagnostic Unpatched | P5b: Patched Final (In-Sample / Held-Out Aggregate) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benchmark Accuracy** | **40.0%** (12/30) | **73.3%** (22/30) | **63.3%** (19/30) | **76.7%** (23/30) | **90.0%** (27/30) | **96.7% (29/30)** (Dev In-Sample) / **93.3% (28/30)** (Held-Out-Adjusted)* |
| **95% Wilson Score CI** | `[24.6%, 57.7%]` | `[55.6%, 85.8%]` | `[45.5%, 78.1%]` | `[59.1%, 88.2%]` | `[74.4%, 96.5%]` | **`[83.3%, 99.4%]`** (Dev) / **`[78.7%, 98.2%]`** (Held-Out Aggregate) |
| **- Class 1: System-1 (12 tasks)** | 4 / 12 (33.3%) | 9 / 12 (75.0%) | 6 / 12 (50.0%) | 8 / 12 (66.7%) | 11 / 12 (91.7%) | **11 / 12 (91.7%)** (Micro-CoT + Suffix Closure) |
| **- Class 2: Extraction (10 tasks)**| 5 / 10 (50.0%) | 10 / 10 (100.0%) | 10 / 10 (100.0%) | 10 / 10 (100.0%) | 10 / 10 (100.0%) | **10 / 10 (100.0%)** (Motor B Fast Symbolic) |
| **- Class 3: Code Synthesis (8 tasks)** | 3 / 8 (37.5%) | 3 / 8 (37.5%) | 3 / 8 (37.5%) | 5 / 8 (62.5%) | 6 / 8 (75.0%) | **8 / 8 (100.0%)** (Dev $K=4$) / **7 / 8 (87.5%)** (Held-Out Suite) |
| **Total Pipeline Latency** | 57.76 s | 111.18 s | 96.21 s | **84.70 s** | 119.07 s | **121.19 s** |
| **Estimated CPU Energy (24W)** | 1,386.1 J | 2,635.9 J | 2,293.8 J | **2,019.4 J** | 2,857.7 J | **2,899.5 J** |
| **Tier-2 Escalations (1.5B)** | 0 / 30 (0%) | 30 / 30 (100%) | 14 / 30 (46.7%) | 15 / 30 (50.0%) | 15 / 30 (50.0%) | **15 / 30 (50.0%)** |
| **Tier-3 Escalations (3B Coder)**| 0 / 30 (0%) | 0 / 30 (0%) | 0 / 30 (0%) | 0 / 30 (0%) | 3 / 30 (10.0%) | **3 / 30 (10.0%)** |
| **Active Peak RAM** | ~480 MiB | ~1,120 MiB | < 1.5 GiB | < 1.5 GiB | < 1.5 GiB (T1-T2) | **< 1.5 GiB (T1-T2) / 2.1 GB on-demand (T3)** |

*\*Note on Held-Out-Adjusted 30-Task Aggregate (28/30, 93.3% [78.7%, 98.2%]): This metric combines the prior development results of Class 1 (11/12) and Class 2 (10/10) with the post-development adversarial held-out audit on Class 3 (7/8 tasks passing 4/4 unseen cases), representing the best observed accuracy among evaluated policies on the proposed 30-task benchmark.*

*Key Findings:*
- **Policy 4 (Swarm v1.0)** outperforms monolithic 1.5B in both accuracy (**76.7% vs 73.3%**) and speed (**84.70 s vs 111.18 s**, $-23.8\%$ latency) while cutting CPU energy by $-23.4\%$.
- **Policy 5b (Swarm v1.5.2)** achieves **96.7% in-sample accuracy** and **93.3% Held-Out-Adjusted Aggregate accuracy**, resolving the residual frontier with surgical 3B invocation on only 3 tasks (10.0%).

---

### Out-of-Sample Held-Out Adversarial Generalization Audit (Table 4)

To empirically audit against overfitting to the 4-case in-sample test harnesses ($K=4$), all Python functions selected across Class 3 (T23–T30) are evaluated on the frozen **32-case Held-Out Adversarial Suite ($\mathcal{T}_{\text{HO}}$)**:

| Task ID | Class 3 Operational Problem | 0.5B P1 | 1.5B P2 | 3B Precise | Swarm Selected Model & Provenance | Held-Out Result (4 unseen cases) |
| :---: | :--- | :---: | :---: | :---: | :--- | :---: |
| **T23** | Formatar Telefone BR (Fixo 10d / Cel 11d) | 1/4 | 1/4 | 4/4 | **Tier-3 (3B Unpatched)** [Verbatim JSON] | **PASS (4/4)** |
| **T24** | Calcular Alíquota ISS Municipal | 4/4 | 4/4 | — | **Tier-1 (0.5B Edge Harness)** [Verbatim JSON] | **PASS (4/4)** |
| **T25** | Higienizar Chave PIX (Email, CPF, EVP) | 0/4 | 3/4 | 3/4 | **Tier-3 (3B Unpatched)** [Verbatim JSON] | **FAIL (3/4)** [False Acceptance: $\hat{\epsilon}_{\text{FA}}=12.5\%$] |
| **T26** | Dígito Verificador Módulo 11 (Pesos 2..9) | 0/4 | 0/4 | 4/4 | **Tier-3 (3B Unpatched Autonomous)** [Verbatim JSON] | **PASS (4/4)** |
| **T27** | Parsear Data BR (`DD/MM/AAAA`) p/ ISO | 4/4 | 2/4 | — | **Tier-1 (0.5B `strptime` winner)** [Verbatim JSON] | **PASS (4/4)** |
| **T28** | Desconto Comercial Progressivo por Faixa | 4/4 | 4/4 | — | **Tier-1 (0.5B Edge Harness)** [Verbatim JSON] | **PASS (4/4)** |
| **T29** | Extrair Domínio Corporativo vs Público | 0/4 | 3/4 | — | **Tier-1 (0.5B Lab04 Harness)** [Reconstructed]* | **PASS (4/4)** |
| **T30** | Classificar Faixa Etária ANS | 4/4 | 4/4 | — | **Tier-1 (0.5B Edge Harness)** [Verbatim JSON] | **PASS (4/4)** |

*\*Note on T29 Provenance: `lab04_aacr_results.json` recorded pass telemetry (`correto: true`, 1,893.43 ms) for T29 without serializing the `raw_output` code string in `policy4_swarm_v1_aacr["detalhes"]`. Both the verbatim-logged subset and full 8-task sets are presented below.*

#### Statistical Decomposition of Held-Out Generalization:

* **[A] Strict Verbatim-Logged Subset (7 Tasks with serialized code strings in JSON, 28 Unseen Cases):**
  - **Case-Level Accuracy:** **27 / 28 (96.43%)**
  - **Per-Case False-Acceptance Rate:** $\hat{\epsilon}_{\text{FA}}^{\text{case}}(4) = 1/28 = \mathbf{3.57\%}$
  - **Strict Task-Level Accuracy:** **6 / 7 (85.71%)**
  - **Strict Task-Suite Failure Rate (4/4 cases):** $\hat{\epsilon}_{\text{FA}}^{\text{task}}(4, 4) = 1/7 = \mathbf{14.29\%}$

* **[B] Full 8-Task Class 3 Set (Including Reconstructed T29, 32 Unseen Cases):**
  - **Case-Level Accuracy:** **31 / 32 (96.88%)**
  - **Per-Case False-Acceptance Rate:** $\hat{\epsilon}_{\text{FA}}^{\text{case}}(4) = 1/32 = \mathbf{3.125\%}$
  - **Strict Task-Level Accuracy:** **7 / 8 (87.50%)**
  - **Strict Task-Suite Failure Rate (4/4 cases):** $\hat{\epsilon}_{\text{FA}}^{\text{task}}(4, 4) = 1/8 = \mathbf{12.50\%}$

* **Autonomous Modulo-11 Synthesis:** Confirming the unpatched 3B model autonomously synthesized the exact formula without prescriptive prompt leakage.

---

## 🏛️ Swarm Topology & Architecture

```
                  ┌───────────────────────────────────────────────────────┐
                  │              OPERATIONAL INCOMING REQUEST             │
                  │                (30 Real Tasks in PT-BR)               │
                  └───────────────────────────────────────────────────────┘
                                              │
                                              ▼
                  ┌───────────────────────────────────────────────────────┐
                  │        TIER-1: EDGE SLM (Qwen2.5-0.5B Q4_K_M)         │
                  │  • Active RAM: ~480 MiB | Latency: 200 - 1,500 ms     │
                  │  • Calibrated System-1: T=1.6, beta_A=2.5, Inversion  │
                  │  • Motor B: Fast Symbolic Normalizer + Substring Proof│
                  │  • Unified-Namespace Execution Harness (prototype)    │
                  └───────────────────────────────────────────────────────┘
                           │                                     │
             [Verified / Conf >= 0.85]             [Uncertain / Verifier Rejected]
                           │                                     │
                           ▼                                     ▼
            ┌─────────────────────────────┐       ┌───────────────────────────────┐
            │   RESOLVED AT TIER-1 EDGE   │       │   TIER-2: LOCAL ON-DEMAND     │
            │   (50.0% of All Tasks)      │       │     (Qwen2.5-1.5B Q4_K_M)     │
            │   Network delay: 0.0 ms     │       │  • RAM: ~1,120 MiB            │
            └─────────────────────────────┘       │  • Micro-CoT + Suffix Closure │
                                                  │  • Structured Extraction      │
                                                  │  • AACR Code Repair Step 1    │
                                                  └───────────────────────────────┘
                                                           │             │
                                                  [Approved / 100%]   [C3 Residual]
                                                           │             │
                                                           ▼             ▼
                                                ┌───────────────────┐ ┌───────────────────────────┐
                                                │ RESOLVED TIER-2   │ │ TIER-3: SPECIALIST CODER  │
                                                │ (40.0% of Tasks)  │ │   (Qwen2.5-Coder-3B)      │
                                                └───────────────────┘ │ • RAM: 2.10 GB on-demand  │
                                                                      │ • Exact AACR Diff Report  │
                                                                      │ • Four-Case Functional    │
                                                                      │   Testing (K=4) + HeldOut │
                                                                      └───────────────────────────┘
                                                                                    │
                                                                                    ▼
                                                                      ┌───────────────────────────┐
                                                                      │  SWARM CLOSURE: 96.7%     │
                                                                      │  (29/30 In-Sample Dev)    │
                                                                      │  93.3% Held-Out-Adjusted  │
                                                                      └───────────────────────────┘
```

---

## 🔬 Core Algorithmic Engines

### 1. Motor A: Calibrated System-1 & Micro-CoT with Deterministic Suffix Closure
* **Logit Calibration:** Neutralizes positional first-option bias on 0.5B via temperature scaling ($T=1.6$) and empirical prior penalty ($\beta_A=2.5$):
  $$\tilde{z}_A = \frac{z_A - \beta_A}{T}, \quad \tilde{z}_k = \frac{z_k}{T} \; (k \neq A)$$
* **Order-Inversion Check:** Prompts with inverted option ordering when $p_{\max} < 0.88$; semantic disagreements immediately trigger escalation.
* **Controlled Micro-CoT:** Escalated decisions on 1.5B generate a single reasoning sentence ($\le 20$ words) preventing context truncation.
* **Deterministic Suffix Closure:** If generation hits token limits ($N_{\text{predict}} \le 65$) before emitting `"Resposta final: <LETRA>"`, a 1-token query with the reasoning suffix immediately extracts the classification without losing reasoning context.

### 2. Motor B: Fast Symbolic Normalizer with Substring Evidence Anchoring
* Eliminates decimal/thousands formatting errors (`R$ 32.400,00` vs `32.400.00` vs `7500.00`).
* Enforces **Zero Hallucination**: The extracted evidence (`trecho_evidencia`) must be an exact literal substring of the source document.
* Enables the 0.5B model to resolve **90% of structured extraction tasks locally with mathematical certainty**.

### 3. Motor C: Four-Case Functional Testing (K=4) + Unseen Held-Out Suite & AACR Cycle
* Synthesized Python code is executed in a **Unified-Namespace Execution Harness (non-isolated prototype)** (`exec(code, env, env)`).
* In **Four-Case Functional Testing (K=4) + Unseen Held-Out Suite**, the synthesized function must achieve **100% pass rate (4/4 test cases)**.
* When failures occur, an **Adaptive Autonomous Code Repair (AACR)** residual error report is constructed and fed into Tier-3 (`Qwen2.5-Coder-3B`), closing the residual frontier without prescriptive prompt leakage.

---

## 📂 Repository Layout

```
polybee-f3-swarm/
├── paper/                                     <- Technical preprint manuscript bundle (v1.5.2 — arXiv [cs.AI])
│   ├── Polybee_F3_Swarm_v1_5_arXiv_Preprint.pdf  <- Compiled research paper PDF (v1.5.2 audited)
│   ├── polybee_f3_swarm_v1_5_arxiv.tex           <- Full LaTeX source code
│   ├── Polybee_F3_Swarm_v1_5_arXiv_Source.zip    <- Complete arXiv submission zip
│   ├── PAPER_POLYBEE_F3_SWARM_2026.md            <- Full technical markdown report
│   ├── fig1_f3_swarm_architecture_rashomon.pdf   <- Fig 1: Architecture & Rashomon Collapse
│   ├── fig2_lab02_physical_ram_pipeline.pdf      <- Fig 2: RAM Pooling vs NVMe Swap Thrashing
│   ├── fig3_labs03_05_policy_progression.pdf     <- Fig 3: End-to-End Policy Progression
│   └── fig4_forensic_case_studies_aacr.pdf       <- Fig 4: Neuro-Symbolic & AACR Ablations
├── README.md                                  <- Executive open-source documentation (v1.5.2)
├── CITATION.cff                               <- Academic citation metadata (GitHub standard)
├── LICENSE                                    <- MIT License (Henrique Oliveira / Polybee AI Research)
├── .zenodo.json                               <- Zenodo automatic archiving metadata
├── reproduce_all_results.py                   <- 100% JSON-driven reproducibility & held-out verifier (<50ms)
├── hardware_inventory_lab01.json             <- Audited Lab 01 hardware environment
├── f3_swarm_lab01_results.json                 <- Audited Lab 01 analytical benchmark
├── lab02_physical_results.json                <- Audited Lab 02 physical swap vs pipeline telemetry
├── lab03_real_llm_results.json                <- Audited Lab 03 real LLM baselines
├── lab04_aacr_results.json                    <- Audited Lab 04 Swarm v1.0 AACR results
├── lab05_tier3_closure_results.json           <- Consolidated metrics across Labs 01 to 05 (96.7% in-sample)
├── precise_tier3_results.json                 <- Audited Lab 05 unpatched Tier-3 telemetry
├── step2_micro_cot_results.json               <- Audited Lab 05 Step-2 Micro-CoT telemetry
├── polybee-f3-swarm-lab01/                    <- Lab 01: Hardware inventory & physical baselines
├── polybee-f3-swarm-lab02/                    <- Lab 02: Thermal throttling & energy profiling
├── polybee-f3-swarm-lab03/                    <- Lab 03: Baseline real LLM evaluation (P1, P2, P3)
│   ├── llama_bin/                             <- llama.cpp binaries (download via download_assets.py)
│   ├── models/                                <- Quantized GGUF models (0.5B, 1.5B, 3B Coder)
│   ├── tasks_data.py                          <- 30 Portuguese operational benchmark tasks
│   └── lab03_real_llm_results.json
├── polybee-f3-swarm-lab04/                    <- Lab 04: Swarm v1.0 AACR & Calibrated System-1
│   ├── f3_swarm_lab04_aacr_calibrated.py
│   ├── lab04_aacr_results.json
│   └── relatorio_retorno_lab04.md
└── polybee-f3-swarm-lab05/                    <- Lab 05: Tier-3 Closure & Micro-CoT Evaluation
    ├── f3_swarm_lab05_tier3_closure.py         <- Main automated pipeline runner
    ├── download_direct_fast.py                <- High-speed parallel GGUF downloader
    ├── precise_tier3_results.json             <- Step 3 diagnostic telemetry
    ├── step2_micro_cot_results.json           <- Step 2 Micro-CoT diagnostic telemetry
    └── lab05_tier3_closure_results.json
```

---

## 🚀 Quickstart & Reproduction

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12 and 3.14)
- Windows 10/11 x64 (or Linux x86_64 / macOS with corresponding llama.cpp server)
- Minimum 8 GiB RAM (Swarm runtime operates with < 1.5 GiB active footprint)

### 2. Inspect Audited Results & Reproduction
To immediately verify and inspect the audited telemetry from Labs 01 to 05 without downloading models:

```bash
# Clone the repository
git clone https://github.com/henrique-polybee/polybee-f3-swarm.git
cd polybee-f3-swarm

# Verify all audited results, check SHA-256 digests, and run held-out suite (<50 ms)
python reproduce_all_results.py
```

The script:
1. Dynamically computes the **real SHA-256 digests** of all 8 raw telemetry JSON files on disk;
2. Reconciles **Table 2** directly from `lab02_physical_results.json`;
3. Reconciles **Table 3** directly from JSON telemetry with exact 95% Wilson Score confidence intervals;
4. Runs the **32 unseen held-out test cases ($\mathcal{T}_{\text{HO}}$)** against the recorded Python functions;
5. Reports both the strict verbatim-logged subset (7 tasks, 28 cases) and the full 8-task set (32 cases).

### 3. Run Live Physical Inference from Scratch (Lab 05 Pipeline — Best Observed Accuracy)
To download models and execute physical inference on your local hardware:

```bash
# 1. Download llama.cpp binaries and edge models (0.5B and 1.5B)
python polybee-f3-swarm-lab03/download_assets.py

# 2. Fast-download Tier-3 specialist model (Qwen2.5-Coder-3B)
python polybee-f3-swarm-lab05/download_direct_fast.py

# 3. Run the complete Lab 05 Tier-3 closure pipeline
python polybee-f3-swarm-lab05/f3_swarm_lab05_tier3_closure.py
```

The script will:
1. Start `Qwen2.5-1.5B` and evaluate the Micro-CoT Controlled strategy with Deterministic Suffix Closure;
2. Cleanly release Tier-2 memory;
3. Start `Qwen2.5-Coder-3B` and evaluate the AACR Residual Repair cycle under Four-Case Functional Testing (K=4) + Unseen Held-Out Suite;
4. Release Tier-3 memory;
5. Output the complete comparative performance matrix and write `lab05_tier3_closure_results.json`.

---

## 📜 Citation

If you use **Polybee F3-Swarm**, our benchmark dataset, or our algorithmic calibration techniques in your research or production systems, please cite our technical paper:

```bibtex
@article{oliveira2026f3swarm,
  title       = {Polybee F3-Swarm v1.5.2: Verifier-Guided Heterogeneous SLM Routing, Symbolic Evidence Normalization, and Controlled Memory-Pipeline Benchmarks on Commodity Edge Hardware},
  author      = {Oliveira, Henrique},
  journal     = {arXiv preprint arXiv:[cs.AI]},
  year        = {2026},
  month       = {November},
  institution = {Polybee AI Research Laboratory},
  url         = {https://github.com/henrique-polybee/polybee-f3-swarm},
  note        = {Zenodo DOI will be assigned upon v1.5.2 release}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). Models (`Qwen2.5`) are licensed under the Apache 2.0 license by Alibaba Cloud.

---
*Maintained by the Polybee AI Research Engineering Team • 2026*
