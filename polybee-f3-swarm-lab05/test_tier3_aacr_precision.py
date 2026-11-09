import os
import sys
import time
import json
import re
import urllib.request
import subprocess

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB03_DIR = os.path.join(WORKSPACE_DIR, "polybee-f3-swarm-lab03")
LAB05_DIR = os.path.dirname(os.path.abspath(__file__))

BIN_DIR = os.path.join(LAB03_DIR, "llama_bin")
SERVER_EXE = os.path.join(BIN_DIR, "llama-server.exe")
MODELS_DIR = os.path.join(LAB03_DIR, "models")
MODEL_3B = os.path.join(MODELS_DIR, "qwen2.5-coder-3b-instruct-q4_k_m.gguf")

sys.path.append(LAB03_DIR)
from tasks_data import CLASSE_3_TASKS

sys.path.append(LAB05_DIR)
from f3_swarm_lab05_tier3_closure import LlamaServerManager, call_completion, verificar_classe_3_rashomon_fixed

PORT = 8089

def main():
    print("Testing Tier-3 Coder 3B with Precise AACR Residual Diagnosis...")
    server = LlamaServerManager(MODEL_3B, port=PORT, ctx_size=1024, threads=4)
    server.start()

    c3_targets = [t for t in CLASSE_3_TASKS if t["id"] in ["T23", "T25", "T26"]]

    aacr_reports = {
        "T23": (
            "- Falha identificada na versão anterior: Para telefone fixo de 10 dígitos (ex: '(21) 3456-7890'), o f-string usou incorretamente 5 dígitos no prefixo.\n"
            "- Correção obrigatória:\n"
            "  * Para 11 dígitos: f'({d[:2]}) {d[2:7]}-{d[7:]}'\n"
            "  * Para 10 dígitos: f'({d[:2]}) {d[2:6]}-{d[6:]}'\n"
            "  * Para qualquer outro tamanho (ex: 5 dígitos): retornar ''."
        ),
        "T25": (
            "- Falha identificada na versão anterior: A função removeu os hífens de chaves aleatórias como '  chave-aleatoria-12345  ', resultando em 'chavealeatoria12345' em vez de 'chave-aleatoria-12345'.\n"
            "- Correção obrigatória:\n"
            "  * Se '@' em chave: retornar chave.strip().lower()\n"
            "  * Limpar apenas caracteres não numéricos com digitos = re.sub(r'\\D', '', chave). Se len(digitos) == 11 e todos os outros caracteres forem pontos ou traços (padrão CPF): retornar digitos.\n"
            "  * Em todos os outros casos: retornar chave.strip() PRESERVANDO os traços internos intactos!"
        ),
        "T26": (
            "- Falhas reportadas pelo verificador funcional (K=4):\n"
            "  * Para entrada ('00000001',), sua função retornou '2', mas o esperado era '9'.\n"
            "  * Para entrada ('12345678',), sua função retornou '2', mas o esperado era '9'."
        )
    }

    results = []
    try:
        for t in c3_targets:
            tid = t["id"]
            exemplos_str = "\n".join([f"- Entrada: {tc['input']} -> Esperado: {repr(tc['output'])}" for tc in t["testes"]])
            relatorio = aacr_reports[tid]

            prompt = (
                f"<|im_start|>system\n"
                f"Você é o especialista de Camada 3 Qwen2.5-Coder-3B da Polybee AI Research.\n"
                f"Você deve implementar a função `def solucao(...)` em Python seguindo estritamente as instruções de correção do Relatório AACR.\n"
                f"Retorne EXCLUSIVAMENTE o código Python dentro de ```python ... ```, sem introdução nem comentários.\n<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Problema: {t['nome']}\n"
                f"{t['descricao']}\n\n"
                f"Casos de validação obrigatórios:\n{exemplos_str}\n\n"
                f"Relatório de Erro Residual AACR:\n{relatorio}\n\n"
                f"Escreva a função `def solucao(...)` 100% correta:\n<|im_end|>\n"
                f"<|im_start|>assistant\n```python\n"
            )

            res = call_completion(prompt, n_predict=350, temperature=0.0)
            raw = "```python\n" + res.get("content", "").strip()
            lat_ms = res["measured_latency_ms"]

            passou, motivo, meta, clean = verificar_classe_3_rashomon_fixed(raw, t["testes"])
            print(f"[{tid}] {t['nome']} -> Passou: {passou} ({motivo}, {meta['acertos']}/{meta['total']}) | Lat: {lat_ms:.1f}ms")
            if not passou:
                for d in meta.get("detalhes", []):
                    if not d.get("ok"):
                        print(f"   Falha caso: inp={d['input']}, obtido={d.get('obtido')}, esp={d.get('esperado')}")
            results.append({"task_id": tid, "passou": passou, "motivo": motivo, "meta": meta, "code": clean, "lat_ms": lat_ms})
    finally:
        server.stop()

    return results

if __name__ == "__main__":
    res = main()
    with open("polybee-f3-swarm-lab05/precise_tier3_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
