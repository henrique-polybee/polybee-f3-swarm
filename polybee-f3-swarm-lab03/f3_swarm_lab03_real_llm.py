"""
Polybee AI Research - Lab 03
Validação Semântica Real com Modelos Pré-Treinados Reais, System-1 Tipado e Verificador Determinístico PT-BR
Modelos:
 - Qwen2.5-0.5B-Instruct-GGUF (Q4_K_M)
 - Qwen2.5-1.5B-Instruct-GGUF (Q4_K_M)
Runtime: llama-server (llama.cpp AVX2/AVX-512)
"""

import os
import sys
import time
import json
import math
import re
import subprocess
import urllib.request
import urllib.error
from typing import Dict, Any, List, Tuple

from tasks_data import (
    CLASSE_1_TASKS,
    CLASSE_2_TASKS,
    CLASSE_3_TASKS,
    verificar_classe_2_evidencia,
    verificar_classe_3_rashomon_loo
)

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(LAB_DIR, "llama_bin")
SERVER_EXE = os.path.join(BIN_DIR, "llama-server.exe")
MODELS_DIR = os.path.join(LAB_DIR, "models")
MODEL_05B = os.path.join(MODELS_DIR, "qwen2.5-0.5b-instruct-q4_k_m.gguf")
MODEL_15B = os.path.join(MODELS_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf")
RESULTS_FILE = os.path.join(LAB_DIR, "lab03_real_llm_results.json")
REPORT_FILE = os.path.join(LAB_DIR, "relatorio_retorno_lab03.md")

PORT = 8089
SERVER_URL = f"http://127.0.0.1:{PORT}"
CPU_POWER_WATTS = 24.0 # Active dynamic CPU power for i7-11370H
REMOTE_NETWORK_DELAY_MS = 45.0 # Simulated remote socket latency for cloud 1.5B pipeline

class LlamaServerManager:
    def __init__(self, model_path: str, port: int = PORT, ctx_size: int = 1024, threads: int = 4):
        self.model_path = model_path
        self.port = port
        self.ctx_size = ctx_size
        self.threads = threads
        self.proc = None
        self.log_file = None

    def start(self):
        log_path = os.path.join(LAB_DIR, f"server_{os.path.basename(self.model_path)}.log")
        self.log_file = open(log_path, "w", encoding="utf-8")
        cmd = [
            SERVER_EXE,
            "-m", self.model_path,
            "--port", str(self.port),
            "-c", str(self.ctx_size),
            "-t", str(self.threads),
            "--log-disable"
        ]
        print(f"[*] Starting llama-server with {os.path.basename(self.model_path)} on port {self.port}...")
        self.proc = subprocess.Popen(cmd, stdout=self.log_file, stderr=subprocess.STDOUT)
        
        # Wait for health
        ready = False
        for _ in range(30):
            time.sleep(0.5)
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{self.port}/health")
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                pass
        if not ready:
            self.stop()
            raise RuntimeError(f"Server failed to start for {self.model_path}!")
        print(f"[+] Server ready (PID: {self.proc.pid})")

    def stop(self):
        if self.proc:
            print(f"[*] Stopping server (PID: {self.proc.pid})...")
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None
        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass
            self.log_file = None
        time.sleep(1.0)
        print("[+] Server stopped and memory released.")

def call_completion(prompt: str, n_predict: int = 128, temperature: float = 0.0, n_probs: int = 0) -> Dict[str, Any]:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature
    }
    if n_probs > 0:
        payload["n_probs"] = n_probs
    
    req = urllib.request.Request(
        f"{SERVER_URL}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    lat_ms = (time.perf_counter() - t0) * 1000
    res["measured_latency_ms"] = lat_ms
    return res

# ==============================================================================
# RUNNERS POR CLASSE
# ==============================================================================

def run_classe_1_traditional(task: Dict[str, Any]) -> Dict[str, Any]:
    """Modo Tradicional Autoregressivo: Força explicação longa + JSON."""
    opcoes_str = "\n".join([f"{k}: {v}" for k, v in task["opcoes"].items()])
    prompt = (
        f"<|im_start|>system\n"
        f"Você é um classificador operacional de triagem e risco em português. "
        f"Analise detalhadamente o caso apresentado, forneça sua justificativa técnica e "
        f"ao final responda estritamente com um bloco JSON no formato:\n"
        f"{{\"justificativa\": \"<sua análise detalhada>\", \"escolha\": \"<LETRA_DA_OPCAO>\"}}\n"
        f"Opções disponíveis:\n{opcoes_str}\n<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Caso: {task['contexto']}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    res = call_completion(prompt, n_predict=150, temperature=0.0)
    raw_text = res.get("content", "").strip()
    
    # Extrai escolha do JSON
    escolha = None
    try:
        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if m:
            dados = json.loads(m.group(0))
            val = dados.get("escolha") or dados.get("resposta") or dados.get("decisao") or ""
            escolha = str(val).strip().upper()
    except Exception:
        pass
    if not escolha:
        # Fallback regex para letra de opção
        valid_keys = "".join(task["opcoes"].keys())
        m2 = re.search(rf"\b([{valid_keys}])\b", raw_text)
        if m2:
            escolha = m2.group(1).upper()

    correto = (escolha == task["gabarito"])
    tokens_gen = res.get("tokens_predicted", 0)
    tokens_prompt = res.get("tokens_evaluated", 0)
    lat_ms = res["measured_latency_ms"]

    return {
        "task_id": task["id"],
        "tipo": "Classe1_Tradicional",
        "nome": task["nome"],
        "escolha": escolha,
        "gabarito": task["gabarito"],
        "correto": correto,
        "tokens_prompt": tokens_prompt,
        "tokens_gen": tokens_gen,
        "latency_ms": lat_ms,
        "raw_output": raw_text[:200]
    }

def run_classe_1_system1(task: Dict[str, Any]) -> Dict[str, Any]:
    """Modo Polybee System-1: 1-token prefill logit probability calculation."""
    opcoes_str = "\n".join([f"{k}: {v}" for k, v in task["opcoes"].items()])
    prompt = (
        f"<|im_start|>system\n"
        f"Você é um classificador operacional ultrarrápido (Polybee System-1). "
        f"Responda estritamente com apenas UMA ÚNICA LETRA dentre as opções, sem introdução nem justificativa.\n"
        f"Opções:\n{opcoes_str}\n<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Caso: {task['contexto']}\n"
        f"Escolha apenas a letra da opção correta:\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    res = call_completion(prompt, n_predict=1, temperature=0.0, n_probs=15)
    gen_token = res.get("content", "").strip().upper()
    lat_ms = res["measured_latency_ms"]
    tokens_prompt = res.get("tokens_evaluated", 0)
    tokens_gen = res.get("tokens_predicted", 1)

    # Extrai logits das opções válidas para calcular softmax calibrado
    valid_options = list(task["opcoes"].keys())
    option_logprobs = {opt: -30.0 for opt in valid_options}
    
    probs_data = res.get("completion_probabilities", [])
    if probs_data and len(probs_data) > 0:
        top_logprobs = probs_data[0].get("top_logprobs", [])
        for entry in top_logprobs:
            t = entry.get("token", "").strip().upper()
            lp = entry.get("logprob", -30.0)
            if t in valid_options and lp > option_logprobs[t]:
                option_logprobs[t] = lp

    # Softmax sobre as opções válidas
    max_lp = max(option_logprobs.values())
    exp_sum = sum(math.exp(lp - max_lp) for lp in option_logprobs.values())
    calibrated_probs = {
        opt: math.exp(option_logprobs[opt] - max_lp) / exp_sum
        for opt in valid_options
    }

    # Melhor opção escolhida e sua confiança
    best_option = max(calibrated_probs, key=calibrated_probs.get)
    # Se o token gerado diretamente for válido, use ele
    if gen_token in valid_options:
        escolha = gen_token
    else:
        escolha = best_option

    confidence = calibrated_probs.get(escolha, 0.0)
    correto = (escolha == task["gabarito"])

    return {
        "task_id": task["id"],
        "tipo": "Classe1_System1",
        "nome": task["nome"],
        "escolha": escolha,
        "gabarito": task["gabarito"],
        "correto": correto,
        "confidence": confidence,
        "calibrated_probs": calibrated_probs,
        "tokens_prompt": tokens_prompt,
        "tokens_gen": tokens_gen,
        "latency_ms": lat_ms,
        "raw_output": gen_token
    }

def run_classe_2_extracao(task: Dict[str, Any]) -> Dict[str, Any]:
    """Classe 2: Extração Estruturada PT-BR avaliada pelo Verificador de Evidência."""
    prompt = (
        f"<|im_start|>system\n"
        f"Você é um extrator de dados de documentos administrativos e financeiros em português. "
        f"Extraia estritamente os campos e responda EXCLUSIVAMENTE com um objeto JSON válido:\n"
        f"{{\n"
        f"  \"cnpj_ou_cpf\": \"<número com pontuação ou limpo>\",\n"
        f"  \"valor_brl\": <valor numérico com ponto decimal>,\n"
        f"  \"categoria\": \"<categoria do serviço>\",\n"
        f"  \"trecho_evidencia\": \"<sub-trecho literal copiado exatamente do texto que comprova o fato>\"\n"
        f"}}\n"
        f"Atenção: 'trecho_evidencia' DEVE SER UMA SUBSTRING EXATA E COPIADA DO TEXTO.\n<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Texto:\n{task['texto']}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    res = call_completion(prompt, n_predict=200, temperature=0.0)
    raw_output = res.get("content", "").strip()
    lat_ms = res["measured_latency_ms"]
    tokens_prompt = res.get("tokens_evaluated", 0)
    tokens_gen = res.get("tokens_predicted", 0)

    # Executa Verificador Determinístico de Evidência
    t_v0 = time.perf_counter()
    aprovado, motivo, dados = verificar_classe_2_evidencia(task["texto"], raw_output, task["gabarito"])
    t_verif_ms = (time.perf_counter() - t_v0) * 1000

    return {
        "task_id": task["id"],
        "tipo": "Classe2_Extracao",
        "nome": task["nome"],
        "aprovado": aprovado,
        "motivo": motivo,
        "dados_extraidos": dados,
        "tokens_prompt": tokens_prompt,
        "tokens_gen": tokens_gen,
        "latency_ms": lat_ms,
        "verifier_latency_ms": t_verif_ms,
        "raw_output": raw_output
    }

def run_classe_3_sintese(task: Dict[str, Any]) -> Dict[str, Any]:
    """Classe 3: Síntese de Regra / Código Python avaliado por Rashomon LOO-CV."""
    exemplos_str = "\n".join([f"- Entrada: {t['input']} -> Esperado: {repr(t['output'])}" for t in task["testes"]])
    prompt = (
        f"<|im_start|>system\n"
        f"Você é um engenheiro de software sênior. Escreva a função em Python com a assinatura `def solucao(...)` "
        f"que resolve o problema especificado. Retorne EXCLUSIVAMENTE o bloco de código Python dentro de ```python ... ```, "
        f"sem explicações nem saudações.\n<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Problema: {task['nome']}\n"
        f"{task['descricao']}\n\n"
        f"Casos de validação:\n{exemplos_str}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    res = call_completion(prompt, n_predict=400, temperature=0.0)
    raw_output = res.get("content", "").strip()
    lat_ms = res["measured_latency_ms"]
    tokens_prompt = res.get("tokens_evaluated", 0)
    tokens_gen = res.get("tokens_predicted", 0)

    # Executa Verificador Rashomon LOO-CV
    t_v0 = time.perf_counter()
    aprovado, motivo, meta_rashomon = verificar_classe_3_rashomon_loo(raw_output, task["testes"])
    t_verif_ms = (time.perf_counter() - t_v0) * 1000

    return {
        "task_id": task["id"],
        "tipo": "Classe3_Sintese",
        "nome": task["nome"],
        "aprovado": aprovado,
        "motivo": motivo,
        "meta_rashomon": meta_rashomon,
        "tokens_prompt": tokens_prompt,
        "tokens_gen": tokens_gen,
        "latency_ms": lat_ms,
        "verifier_latency_ms": t_verif_ms,
        "raw_output": raw_output[:500]
    }

# ==============================================================================
# BENCHMARK PRINCIPAL
# ==============================================================================

def main():
    print("=" * 80)
    print("POLYBEE RESEARCH LAB 03 - BENCHMARK SEMÂNTICO REAL (30 TAREFAS PT-BR)")
    print("Modelos: Qwen2.5-0.5B-Instruct & Qwen2.5-1.5B-Instruct (GGUF Q4_K_M)")
    print("Hardware: Intel Core i7-11370H (TDP ~35W, 24W dinâmico)")
    print("=" * 80)

    results = {
        "metadata": {
            "lab": "Lab 03 - Real LLM Semantic & System-1 Validation",
            "cpu": "Intel Core i7-11370H",
            "threads": 4,
            "models": {
                "small": "Qwen2.5-0.5B-Instruct-GGUF (Q4_K_M)",
                "large": "Qwen2.5-1.5B-Instruct-GGUF (Q4_K_M)"
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_tasks": 30
        },
        "system1_comparison_classe1": [],
        "policy1_small_always": {},
        "policy2_large_always": {},
        "policy3_cascaded_swarm": {}
    }

    # --------------------------------------------------------------------------
    # FASE 1: EXECUÇÃO DO MODELO RÁPIDO (0.5B)
    # --------------------------------------------------------------------------
    server_05b = LlamaServerManager(MODEL_05B, port=PORT, ctx_size=1024, threads=4)
    server_05b.start()

    print("\n>>> FASE 1.1: Classe 1 no 0.5B (Comparação Tradicional vs System-1)")
    classe1_05b_trad = []
    classe1_05b_sys1 = []
    for t in CLASSE_1_TASKS:
        res_trad = run_classe_1_traditional(t)
        res_sys1 = run_classe_1_system1(t)
        classe1_05b_trad.append(res_trad)
        classe1_05b_sys1.append(res_sys1)
        print(f"  [{t['id']}] {t['nome'][:30]:<30} | Trad: {res_trad['latency_ms']:6.1f}ms (OK={res_trad['correto']}) | Sys-1: {res_sys1['latency_ms']:5.1f}ms (OK={res_sys1['correto']}, Conf={res_sys1['confidence']*100:4.1f}%)")
        results["system1_comparison_classe1"].append({
            "task_id": t["id"],
            "nome": t["nome"],
            "gabarito": t["gabarito"],
            "traditional_05b": res_trad,
            "system1_05b": res_sys1
        })

    print("\n>>> FASE 1.2: Classe 2 no 0.5B (Extração Estruturada + Verificador de Evidência)")
    classe2_05b = []
    for t in CLASSE_2_TASKS:
        res_c2 = run_classe_2_extracao(t)
        classe2_05b.append(res_c2)
        print(f"  [{t['id']}] {t['nome'][:30]:<30} | {res_c2['latency_ms']:6.1f}ms | Aprovado={res_c2['aprovado']} ({res_c2['motivo']})")

    print("\n>>> FASE 1.3: Classe 3 no 0.5B (Síntese de Regra + Rashomon LOO-CV)")
    classe3_05b = []
    for t in CLASSE_3_TASKS:
        res_c3 = run_classe_3_sintese(t)
        classe3_05b.append(res_c3)
        print(f"  [{t['id']}] {t['nome'][:30]:<30} | {res_c3['latency_ms']:6.1f}ms | Aprovado={res_c3['aprovado']} ({res_c3['motivo']})")

    server_05b.stop()

    # Identificar quais tarefas na Política 3 precisam de escalada:
    # - Classe 1: Confiança System-1 < 0.75
    # - Classe 2: Não aprovado pelo Verificador de Evidência
    # - Classe 3: Não aprovado pelo Verificador Rashomon
    escalation_needed = {
        "classe1": [r["task_id"] for r in classe1_05b_sys1 if r["confidence"] < 0.75],
        "classe2": [r["task_id"] for r in classe2_05b if not r["aprovado"]],
        "classe3": [r["task_id"] for r in classe3_05b if not r["aprovado"]]
    }
    total_escalations = len(escalation_needed["classe1"]) + len(escalation_needed["classe2"]) + len(escalation_needed["classe3"])
    print(f"\n[+] Análise Preliminar do Swarm: {total_escalations}/30 tarefas marcadas para escalada ao 1.5B:")
    print(f"    - Classe 1 (Conf < 75%): {escalation_needed['classe1']}")
    print(f"    - Classe 2 (Falha Evidência): {escalation_needed['classe2']}")
    print(f"    - Classe 3 (Falha Rashomon): {escalation_needed['classe3']}")

    # --------------------------------------------------------------------------
    # FASE 2: EXECUÇÃO DO MODELO ESCALADO (1.5B)
    # --------------------------------------------------------------------------
    server_15b = LlamaServerManager(MODEL_15B, port=PORT, ctx_size=1024, threads=4)
    server_15b.start()

    print("\n>>> FASE 2.1: Executando todas as 30 tarefas no 1.5B (Política 2: Large Always)")
    classe1_15b_trad = []
    for t in CLASSE_1_TASKS:
        res = run_classe_1_traditional(t)
        # Adiciona delay de rede simulado de pipeline remoto
        res["remote_network_latency_ms"] = REMOTE_NETWORK_DELAY_MS
        res["total_pipeline_ms"] = res["latency_ms"] + REMOTE_NETWORK_DELAY_MS
        classe1_15b_trad.append(res)
        print(f"  [{t['id']}] {t['nome'][:30]:<30} | {res['latency_ms']:6.1f}ms + {REMOTE_NETWORK_DELAY_MS}ms rede (OK={res['correto']})")

    classe2_15b = []
    for t in CLASSE_2_TASKS:
        res = run_classe_2_extracao(t)
        res["remote_network_latency_ms"] = REMOTE_NETWORK_DELAY_MS
        res["total_pipeline_ms"] = res["latency_ms"] + REMOTE_NETWORK_DELAY_MS
        classe2_15b.append(res)
        print(f"  [{t['id']}] {t['nome'][:30]:<30} | {res['latency_ms']:6.1f}ms + {REMOTE_NETWORK_DELAY_MS}ms rede | Aprovado={res['aprovado']}")

    classe3_15b = []
    for t in CLASSE_3_TASKS:
        res = run_classe_3_sintese(t)
        res["remote_network_latency_ms"] = REMOTE_NETWORK_DELAY_MS
        res["total_pipeline_ms"] = res["latency_ms"] + REMOTE_NETWORK_DELAY_MS
        classe3_15b.append(res)
        print(f"  [{t['id']}] {t['nome'][:30]:<30} | {res['latency_ms']:6.1f}ms + {REMOTE_NETWORK_DELAY_MS}ms rede | Aprovado={res['aprovado']}")

    server_15b.stop()

    # --------------------------------------------------------------------------
    # FASE 3: MONTAGEM E CONSOLIDAÇÃO DAS POLÍTICAS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CONSOLIDAÇÃO DOS RESULTADOS COMPARATIVOS")
    print("=" * 80)

    # 1. Política 1: Small Always (0.5B para tudo)
    p1_tasks = classe1_05b_trad + classe2_05b + classe3_05b
    p1_total_time_ms = sum(t["latency_ms"] for t in p1_tasks)
    p1_acertos_c1 = sum(1 for t in classe1_05b_trad if t["correto"])
    p1_acertos_c2 = sum(1 for t in classe2_05b if t["aprovado"])
    p1_acertos_c3 = sum(1 for t in classe3_05b if t["aprovado"])
    p1_acertos_total = p1_acertos_c1 + p1_acertos_c2 + p1_acertos_c3
    p1_acc = p1_acertos_total / 30.0
    p1_joules = (p1_total_time_ms / 1000.0) * CPU_POWER_WATTS

    results["policy1_small_always"] = {
        "nome": "Small_Always_0.5B",
        "acertos_classe1": f"{p1_acertos_c1}/12",
        "acertos_classe2": f"{p1_acertos_c2}/10",
        "acertos_classe3": f"{p1_acertos_c3}/8",
        "acertos_total": f"{p1_acertos_total}/30",
        "acuracia_global": p1_acc,
        "tempo_total_segundos": p1_total_time_ms / 1000.0,
        "media_ms_por_tarefa": p1_total_time_ms / 30.0,
        "energia_estimada_joules": p1_joules,
        "detalhes": p1_tasks
    }

    # 2. Política 2: Large Always (1.5B para tudo + overhead remoto)
    p2_tasks = classe1_15b_trad + classe2_15b + classe3_15b
    p2_raw_compute_ms = sum(t["latency_ms"] for t in p2_tasks)
    p2_total_time_ms = sum(t["total_pipeline_ms"] for t in p2_tasks)
    p2_acertos_c1 = sum(1 for t in classe1_15b_trad if t["correto"])
    p2_acertos_c2 = sum(1 for t in classe2_15b if t["aprovado"])
    p2_acertos_c3 = sum(1 for t in classe3_15b if t["aprovado"])
    p2_acertos_total = p2_acertos_c1 + p2_acertos_c2 + p2_acertos_c3
    p2_acc = p2_acertos_total / 30.0
    p2_joules = (p2_raw_compute_ms / 1000.0) * CPU_POWER_WATTS

    results["policy2_large_always"] = {
        "nome": "Large_Always_1.5B",
        "acertos_classe1": f"{p2_acertos_c1}/12",
        "acertos_classe2": f"{p2_acertos_c2}/10",
        "acertos_classe3": f"{p2_acertos_c3}/8",
        "acertos_total": f"{p2_acertos_total}/30",
        "acuracia_global": p2_acc,
        "tempo_computacao_local_ms": p2_raw_compute_ms,
        "tempo_total_pipeline_segundos": p2_total_time_ms / 1000.0,
        "media_ms_por_tarefa": p2_total_time_ms / 30.0,
        "energia_estimada_joules": p2_joules,
        "detalhes": p2_tasks
    }

    # 3. Política 3: Polybee F3 Cascaded Swarm
    p3_tasks = []
    p3_total_time_ms = 0.0
    p3_compute_time_ms = 0.0
    p3_escalations_count = 0

    # Classe 1 Swarm (System-1 -> Escala se confiança < 0.75)
    p3_c1_acertos = 0
    c1_15b_map = {t["task_id"]: t for t in classe1_15b_trad}
    for t_s1 in classe1_05b_sys1:
        tid = t_s1["task_id"]
        if t_s1["confidence"] >= 0.75:
            # Resolvido no System-1 ultrarrápido
            p3_tasks.append({
                "task_id": tid,
                "classe": 1,
                "resolvido_em": "0.5B_System1",
                "escalado": False,
                "correto": t_s1["correto"],
                "latency_ms": t_s1["latency_ms"],
                "confidence": t_s1["confidence"],
                "calibrated_probs": t_s1["calibrated_probs"]
            })
            p3_total_time_ms += t_s1["latency_ms"]
            p3_compute_time_ms += t_s1["latency_ms"]
            if t_s1["correto"]:
                p3_c1_acertos += 1
        else:
            # Escalado para 1.5B
            p3_escalations_count += 1
            t_15b = c1_15b_map[tid]
            combined_lat = t_s1["latency_ms"] + t_15b["total_pipeline_ms"]
            p3_tasks.append({
                "task_id": tid,
                "classe": 1,
                "resolvido_em": "1.5B_Escalada",
                "escalado": True,
                "motivo_escalada": f"Baixa confiança System-1 ({t_s1['confidence']*100:.1f}% < 75%)",
                "correto": t_15b["correto"],
                "latency_ms": combined_lat,
                "sys1_confidence": t_s1["confidence"]
            })
            p3_total_time_ms += combined_lat
            p3_compute_time_ms += (t_s1["latency_ms"] + t_15b["latency_ms"])
            if t_15b["correto"]:
                p3_c1_acertos += 1

    # Classe 2 Swarm (0.5B + Verificador de Evidência -> Escala se reprovar)
    p3_c2_acertos = 0
    c2_15b_map = {t["task_id"]: t for t in classe2_15b}
    for t_05b in classe2_05b:
        tid = t_05b["task_id"]
        if t_05b["aprovado"]:
            # Aprovado com prova matemática pelo Verificador Estrutural
            p3_tasks.append({
                "task_id": tid,
                "classe": 2,
                "resolvido_em": "0.5B_VerificadorAprovou",
                "escalado": False,
                "correto": True,
                "latency_ms": t_05b["latency_ms"] + t_05b["verifier_latency_ms"]
            })
            p3_total_time_ms += (t_05b["latency_ms"] + t_05b["verifier_latency_ms"])
            p3_compute_time_ms += t_05b["latency_ms"]
            p3_c2_acertos += 1
        else:
            # Verificador reprovou (ex: evidência inválida ou tipo errado) -> Escala para 1.5B
            p3_escalations_count += 1
            t_15b = c2_15b_map[tid]
            combined_lat = t_05b["latency_ms"] + t_05b["verifier_latency_ms"] + t_15b["total_pipeline_ms"]
            p3_tasks.append({
                "task_id": tid,
                "classe": 2,
                "resolvido_em": "1.5B_Escalada",
                "escalado": True,
                "motivo_escalada": f"Reprovado pelo Verificador ({t_05b['motivo']})",
                "correto": t_15b["aprovado"],
                "latency_ms": combined_lat
            })
            p3_total_time_ms += combined_lat
            p3_compute_time_ms += (t_05b["latency_ms"] + t_15b["latency_ms"])
            if t_15b["aprovado"]:
                p3_c2_acertos += 1

    # Classe 3 Swarm (0.5B + Verificador Rashomon LOO-CV -> Escala se reprovar)
    p3_c3_acertos = 0
    c3_15b_map = {t["task_id"]: t for t in classe3_15b}
    for t_05b in classe3_05b:
        tid = t_05b["task_id"]
        if t_05b["aprovado"]:
            # Passou com 100% no Rashomon LOO-CV
            p3_tasks.append({
                "task_id": tid,
                "classe": 3,
                "resolvido_em": "0.5B_Rashomon100Pct",
                "escalado": False,
                "correto": True,
                "latency_ms": t_05b["latency_ms"] + t_05b["verifier_latency_ms"]
            })
            p3_total_time_ms += (t_05b["latency_ms"] + t_05b["verifier_latency_ms"])
            p3_compute_time_ms += t_05b["latency_ms"]
            p3_c3_acertos += 1
        else:
            # Falhou no Rashomon -> Escala para 1.5B
            p3_escalations_count += 1
            t_15b = c3_15b_map[tid]
            combined_lat = t_05b["latency_ms"] + t_05b["verifier_latency_ms"] + t_15b["total_pipeline_ms"]
            p3_tasks.append({
                "task_id": tid,
                "classe": 3,
                "resolvido_em": "1.5B_Escalada",
                "escalado": True,
                "motivo_escalada": f"Reprovado no Rashomon LOO-CV ({t_05b['motivo']})",
                "correto": t_15b["aprovado"],
                "latency_ms": combined_lat
            })
            p3_total_time_ms += combined_lat
            p3_compute_time_ms += (t_05b["latency_ms"] + t_15b["latency_ms"])
            if t_15b["aprovado"]:
                p3_c3_acertos += 1

    p3_acertos_total = p3_c1_acertos + p3_c2_acertos + p3_c3_acertos
    p3_acc = p3_acertos_total / 30.0
    p3_joules = (p3_compute_time_ms / 1000.0) * CPU_POWER_WATTS
    escalation_avoidance_rate = (30 - p3_escalations_count) / 30.0

    results["policy3_cascaded_swarm"] = {
        "nome": "Polybee_F3_Cascaded_Swarm",
        "acertos_classe1": f"{p3_c1_acertos}/12",
        "acertos_classe2": f"{p3_c2_acertos}/10",
        "acertos_classe3": f"{p3_c3_acertos}/8",
        "acertos_total": f"{p3_acertos_total}/30",
        "acuracia_global": p3_acc,
        "tempo_total_pipeline_segundos": p3_total_time_ms / 1000.0,
        "media_ms_por_tarefa": p3_total_time_ms / 30.0,
        "energia_estimada_joules": p3_joules,
        "tarefas_escaladas": p3_escalations_count,
        "taxa_escalada_evitada": escalation_avoidance_rate,
        "detalhes": p3_tasks
    }

    # Salva arquivo JSON
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Resultados salvos com sucesso em: {RESULTS_FILE}")

    # Imprime Tabela Resumo
    print("\n" + "=" * 80)
    print("QUADRO RESUMO DE PERFORMANCE E ACURÁCIA (LAB 03)")
    print("=" * 80)
    print(f"{'Métrica':<32} | {'Small Always (0.5B)':<20} | {'Large Always (1.5B)':<20} | {'Polybee F3 Swarm':<20}")
    print("-" * 100)
    print(f"{'Acurácia Global':<32} | {p1_acc*100:6.1f}% ({p1_acertos_total}/30)       | {p2_acc*100:6.1f}% ({p2_acertos_total}/30)       | {p3_acc*100:6.1f}% ({p3_acertos_total}/30)")
    print(f"{'  - Classe 1 (System-1)':<32} | {p1_acertos_c1}/12                  | {p2_acertos_c1}/12                  | {p3_c1_acertos}/12")
    print(f"{'  - Classe 2 (Extração)':<32} | {p1_acertos_c2}/10                  | {p2_acertos_c2}/10                  | {p3_c2_acertos}/10")
    print(f"{'  - Classe 3 (Síntese LOO)':<32} | {p1_acertos_c3}/8                   | {p2_acertos_c3}/8                   | {p3_c3_acertos}/8")
    print(f"{'Tempo Total de Execução':<32} | {p1_total_time_ms/1000.0:6.2f} s              | {p2_total_time_ms/1000.0:6.2f} s              | {p3_total_time_ms/1000.0:6.2f} s")
    print(f"{'Latência Média por Tarefa':<32} | {p1_total_time_ms/30.0:6.1f} ms             | {p2_total_time_ms/30.0:6.1f} ms             | {p3_total_time_ms/30.0:6.1f} ms")
    print(f"{'Energia Estimada (CPU)':<32} | {p1_joules:6.1f} J              | {p2_joules:6.1f} J              | {p3_joules:6.1f} J")
    print(f"{'Taxa de Escalada Evitada':<32} | {'0% (Preso no 0.5B)':<20} | {'0% (Gasta 1.5B sempre)':<20} | {escalation_avoidance_rate*100:6.1f}% ({30-p3_escalations_count}/30)")
    print("=" * 100)

if __name__ == "__main__":
    main()
