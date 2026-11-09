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
        self.proc = subprocess.Popen(cmd, cwd=BIN_DIR, stdout=self.log_file, stderr=subprocess.STDOUT)
        for _ in range(40):
            time.sleep(0.5)
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{self.port}/health")
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                pass

    def stop(self):
        if self.proc:
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

def call_completion(prompt: str, n_predict: int = 65, temperature: float = 0.0) -> dict:
    payload = {"prompt": prompt, "n_predict": n_predict, "temperature": temperature}
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

def test_suffix():
    server = LlamaServerManager(MODEL_15B)
    server.start()
    
    target_tasks = [t for t in CLASSE_1_TASKS if t["id"] in ["T01", "T03", "T08", "T09"]]
    results = []
    
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
            
            res = call_completion(prompt, n_predict=65, temperature=0.0)
            raw = res.get("content", "").strip()
            lat_ms = res["measured_latency_ms"]
            
            suffix_triggered = False
            m_choice = re.search(r"Resposta final:\s*([A-E])", raw, re.IGNORECASE)
            if not m_choice:
                # Dispara Fechamento Determinístico de Sufixo
                suffix_triggered = True
                suffix_prompt = prompt + raw.strip() + "\nResposta final: "
                res_suffix = call_completion(suffix_prompt, n_predict=1, temperature=0.0)
                tok = res_suffix.get("content", "").strip().upper()
                m_sub = re.findall(r"\b([A-E])\b", tok)
                letter = m_sub[0] if m_sub else tok[:1]
                lat_ms += res_suffix["measured_latency_ms"]
                raw = raw.strip() + f"\nResposta final: {letter}"
                escolha = letter
            else:
                escolha = m_choice.group(1).upper()
                
            partes = raw.split("Resposta final:")
            raciocinio = partes[0].strip()
            correto = (escolha == t["gabarito"])
            
            print(f"[{tid}] {t['nome']} | Gabarito: {t['gabarito']} | Escolha: {escolha} | Correto: {correto} | Suffix: {suffix_triggered} | Lat: {lat_ms:.1f}ms")
            print(f"   Raciocínio: {raciocinio}")
            results.append({
                "task_id": tid,
                "gabarito": t["gabarito"],
                "escolha": escolha,
                "correto": correto,
                "suffix_triggered": suffix_triggered,
                "raciocinio": raciocinio,
                "raw_output": raw,
                "lat_ms": lat_ms
            })
    finally:
        server.stop()
        
    return results

if __name__ == "__main__":
    res = test_suffix()
    with open("polybee-f3-swarm-lab05/suffix_closure_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
