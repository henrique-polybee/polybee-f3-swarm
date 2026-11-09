import os
import sys
import time
import json
import re
import urllib.request
import subprocess

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB03_DIR = os.path.join(WORKSPACE_DIR, "polybee-f3-swarm-lab03")
LAB04_DIR = os.path.join(WORKSPACE_DIR, "polybee-f3-swarm-lab04")
LAB05_DIR = os.path.dirname(os.path.abspath(__file__))

BIN_DIR = os.path.join(LAB03_DIR, "llama_bin")
SERVER_EXE = os.path.join(BIN_DIR, "llama-server.exe")
MODELS_DIR = os.path.join(LAB03_DIR, "models")
MODEL_15B = os.path.join(MODELS_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf")

sys.path.append(LAB03_DIR)
from tasks_data import CLASSE_1_TASKS

PORT = 8089
SERVER_URL = f"http://127.0.0.1:{PORT}"

class LlamaServerManager:
    def __init__(self, model_path: str, port: int = PORT, ctx_size: int = 1024, threads: int = 4):
        self.model_path = model_path
        self.port = port
        self.ctx_size = ctx_size
        self.threads = threads
        self.proc = None
        self.log_file = None

    def start(self):
        log_path = os.path.join(LAB05_DIR, f"server_{os.path.basename(self.model_path)}.log")
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
        self.proc = subprocess.Popen(cmd, cwd=BIN_DIR, stdout=self.log_file, stderr=subprocess.STDOUT)
        
        ready = False
        for _ in range(40):
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

def call_completion(prompt: str, n_predict: int = 45, temperature: float = 0.0) -> dict:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature
    }
    req = urllib.request.Request(
        f"{SERVER_URL}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    res["measured_latency_ms"] = (time.perf_counter() - t0) * 1000
    return res

def run_step2():
    print("=" * 80)
    print("LAB 05 - ETAPA 2: SYSTEM-1 COM MICRO-COT CONTROLADO NO QWEN2.5-1.5B")
    print("=" * 80)
    
    server = LlamaServerManager(MODEL_15B, port=PORT)
    server.start()
    
    target_ids = ["T01", "T03", "T08", "T09"]
    target_tasks = [t for t in CLASSE_1_TASKS if t["id"] in target_ids]
    
    # Lab 04 1-token pure baselines for comparison
    with open(os.path.join(LAB04_DIR, "lab04_aacr_results.json"), "r", encoding="utf-8") as f:
        lab04_data = json.load(f)
    lab04_c1_map = {d["task_id"]: d for d in lab04_data["policy4_swarm_v1_aacr"]["detalhes"] if d["task_id"] in target_ids}
    
    results_step2 = []
    
    try:
        for t in target_tasks:
            tid = t["id"]
            opcoes_str = "\n".join([f"{k}: {v}" for k, v in t["opcoes"].items()])
            
            prompt = (
                f"<|im_start|>system\n"
                f"Você é um classificador operacional de triagem e risco em português (Polybee System-1 Micro-CoT).\n"
                f"Analise o caso e responda estritamente no seguinte padrão de 2 partes:\n"
                f"<raciocinio_curto de até 20 palavras>\n"
                f"Resposta final: <LETRA>\n"
                f"Onde <LETRA> é uma das opções válidas ({', '.join(t['opcoes'].keys())}). Não adicione nada mais.\n<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Opções disponíveis:\n{opcoes_str}\n\n"
                f"Caso: {t['contexto']}\n\n"
                f"Apresente seu raciocínio curto de até 20 palavras e a resposta final:\n<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            res = call_completion(prompt, n_predict=45, temperature=0.0)
            raw = res.get("content", "").strip()
            lat_ms = res["measured_latency_ms"]
            
            # Extração de raciocínio curto e escolha
            m_choice = re.search(r"Resposta final:\s*([A-E])", raw, re.IGNORECASE)
            if not m_choice:
                # Fallback: última letra identificada
                m_letters = re.findall(r"\b([A-E])\b", raw)
                escolha = m_letters[-1].upper() if m_letters else "N/A"
            else:
                escolha = m_choice.group(1).upper()
            
            # Extrair frase de raciocínio
            partes = raw.split("Resposta final:")
            raciocinio = partes[0].strip() if len(partes) > 1 else raw
            raciocinio = re.sub(r"\n+", " ", raciocinio).strip()
            palavras_raciocinio = len(raciocinio.split())
            
            correto = (escolha == t["gabarito"])
            
            lab04_baseline = lab04_c1_map.get(tid, {})
            lab04_lat = lab04_baseline.get("latency_ms", 0.0)
            
            item = {
                "task_id": tid,
                "nome": t["nome"],
                "gabarito": t["gabarito"],
                "raw_output": raw,
                "raciocinio_curto": raciocinio,
                "palavras_raciocinio": palavras_raciocinio,
                "escolha_micro_cot": escolha,
                "correto_micro_cot": correto,
                "latencia_micro_cot_ms": lat_ms,
                "lab04_1token_correto": lab04_baseline.get("correto", False),
                "lab04_1token_latencia_ms": lab04_lat,
                "delta_latencia_ms": lat_ms - lab04_lat
            }
            results_step2.append(item)
            
            print(f"\n--- [{tid}] {t['nome']} ---")
            print(f"Gabarito: {t['gabarito']} ({t.get('gabarito_label')})")
            print(f"Raw Output: {raw!r}")
            print(f"Raciocínio ({palavras_raciocinio} palavras): {raciocinio}")
            print(f"Escolha: {escolha} | Correto: {correto}")
            print(f"Latência Micro-CoT: {lat_ms:.1f} ms vs Lab04 1-Token: {lab04_lat:.1f} ms")
            
    finally:
        server.stop()
        
    return results_step2

if __name__ == "__main__":
    res = run_step2()
    with open(os.path.join(LAB05_DIR, "step2_micro_cot_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("\n[+] Step 2 results saved to step2_micro_cot_results.json")
