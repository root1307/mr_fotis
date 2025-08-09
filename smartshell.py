import os
import sys
import json
import time
import queue
import shlex
import ctypes
import threading
import subprocess
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, scrolledtext

def _pip_install(pkg: str):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                               "--break-system-packages", pkg])
    except subprocess.CalledProcessError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])


def _ensure_dependencies(packages):
    """Ensure that required Python packages are installed. Uses pip to install missing packages automatically."""
    for pkg in packages:
        try:
            __import__(pkg)
        except ModuleNotFoundError:
            _pip_install(pkg)



_ensure_dependencies(["llama_cpp", "diskcache"])


LLAMA_AVAILABLE = False
try:
    from llama_cpp import Llama 
    LLAMA_AVAILABLE = True
except Exception:
    Llama = None  
    LLAMA_AVAILABLE = False




def is_frozen() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def app_dir() -> Path:
    if is_frozen():
        return Path(os.path.dirname(sys.executable))
    return Path(__file__).resolve().parent


def bundle_dir() -> Path:
    return Path(getattr(sys, "_MEIPASS", str(app_dir())))


def user_config_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", app_dir()))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    d = base / "SmartShellAI"
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_logs_dir() -> Path:
    d = user_config_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Settings:
    model_name: str = "wizardcoder-python-7b-v1.0.Q4_K_M.gguf"
    ctx: int = 2048
    timeout_sec: int = 600  
    use_shell: bool = True 
    os_hint: str = ""       


SETTINGS = Settings()


MODEL_URL = (
    "https://huggingface.co/TheBloke/WizardCoder-Python-7B-V1.0-GGUF/"
    "resolve/main/wizardcoder-python-7b-v1.0.Q4_K_M.gguf"
)




def wire_llama_lib():
    """Discover the llama-cpp shared library so the import doesn't crash. It's ok if this fails; we have fallback rules."""
    if os.environ.get("LLAMA_CPP_LIB"):
        return
    candidates = [
        bundle_dir() / "llama.dll",
        bundle_dir() / "libllama.so",
        bundle_dir() / "libllama.dylib",
        bundle_dir() / "llama_cpp" / "lib" / "llama.dll",
        bundle_dir() / "llama_cpp" / "lib" / "libllama.so",
        bundle_dir() / "llama_cpp" / "lib" / "libllama.dylib",
    ]
    for p in candidates:
        if p.exists():
            os.environ["LLAMA_CPP_LIB"] = str(p)
            if os.name == "nt" and hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(p.parent))
                except Exception:
                    pass
            return


wire_llama_lib()




def models_dir() -> Path:
    d = app_dir() / "models"
    if d.is_dir():
        return d
    d = user_config_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_path() -> Path:
    return models_dir() / SETTINGS.model_name


def download_model() -> bool:
    """
    Download the GGUF model file if it does not already exist.

    Returns True on success, False on failure. Uses urllib to avoid extra deps.
    """
    dest = model_path()
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request

        with urllib.request.urlopen(MODEL_URL) as response, open(dest, "wb") as out_file:
            total_size = int(response.getheader("Content-Length", 0) or 0)
            downloaded = 0
            chunk_size = 8192
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
        return True
    except Exception as ex:
        try:
            messagebox.showerror(
                "SmartShell AI",
                f"Failed to download model from {MODEL_URL}:\n{ex}",
            )
        except Exception:
            pass
        return False




_LLM = None
_LLM_LOCK = threading.Lock()
TRANSLATE_MODE = "rule"  

def _current_os_hint() -> str:
    if SETTINGS.os_hint:
        return SETTINGS.os_hint
    return "windows" if os.name == "nt" else "linux"

SYSTEM_PROMPT = (
    "You are SmartShell, a terminal assistant. Output ONLY a single-line command suitable for the user's OS.\n"
    "Rules:\n"
    "- No explanations, no backticks, no comments.\n"
    "- Prefer safe flags. If OS is Windows, output a PowerShell command. Otherwise, bash.\n"
    f"- Current OS hint: {_current_os_hint()}.\n\n"
    "Examples:\n"
    "User: update the system\nCommand: sudo apt update && sudo apt upgrade -y\n"
    "User: install vlc\nCommand: sudo apt install -y vlc\n"
    "User: check my IP address\nCommand: ip a\n"
    "User: list current directory\nCommand: ls -la\n"
)


def ensure_model_loaded():
    global _LLM
    if not LLAMA_AVAILABLE:
        raise RuntimeError("llama_cpp is not available. Using fallback rules.")
    with _LLM_LOCK:
        if _LLM is None:
            p = model_path()
            if not p.exists():

                ok = download_model()
                if not ok or not p.exists():
                    raise FileNotFoundError(
                        f"Model missing and download failed: {p}\n"
                        f"Place the .gguf model in '{models_dir()}' or check internet connection."
                    )
            _LLM = Llama(model_path=str(p), n_ctx=SETTINGS.ctx)


def llm_translate(user_prompt: str) -> str:
    global TRANSLATE_MODE
    ensure_model_loaded()
    assert _LLM is not None
    text = f"{SYSTEM_PROMPT}\nUser: {user_prompt}\nCommand:"
    out = _LLM(
        prompt=text,
        max_tokens=120,
        stop=["\n", "</s>", "User:", "Command:"],
        temperature=0.2,
        top_p=0.95,
    )
    cmd = (out.get("choices", [{}])[0].get("text") or "").strip()

    cmd = cmd.splitlines()[0].strip()
    if cmd.startswith("`") and cmd.endswith("`"):
        cmd = cmd[1:-1].strip()
    TRANSLATE_MODE = "llm"
    return cmd


def rule_based_translate(prompt: str) -> str:
    global TRANSLATE_MODE
    low = prompt.strip().lower()
    rules = {
        "κάνε update": "sudo apt update && sudo apt upgrade -y",
        "κανε update": "sudo apt update && sudo apt upgrade -y",
        "άδειασε την cache": "sudo apt clean",
        "αδειασε την cache": "sudo apt clean",
        "ip": "ip a",
        "upgrade": "sudo apt update && sudo apt upgrade -y",
        "install vlc": "sudo apt install -y vlc",
        "list files": "ls -la",
        "ping google": "ping -c 4 google.com",
        "check my ip": "ip a",
        "my ip": "ip a",
    }
    for k, v in rules.items():
        if k in low:
            TRANSLATE_MODE = "rule"
            return v
    TRANSLATE_MODE = "rule"
    return "echo 'No known mapping; please refine your request.'"


def translate(prompt: str) -> str:
    if not prompt.strip():
        return ""
    if LLAMA_AVAILABLE:
        try:
            return llm_translate(prompt)
        except Exception:
            return rule_based_translate(prompt)
    else:
        return rule_based_translate(prompt)





class Runner:
    def __init__(self, timeout_sec: int = 600, use_shell: bool = True):
        self.timeout = timeout_sec
        self.use_shell = use_shell
        self.proc: subprocess.Popen | None = None
        self.output_q: "queue.Queue[str]" = queue.Queue()
        self._cancel = threading.Event()

    def run_async(self, cmd: str, on_done):
        t = threading.Thread(target=self._worker, args=(cmd, on_done), daemon=True)
        t.start()

    def cancel(self):
        self._cancel.set()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                time.sleep(0.7)
                if self.proc.poll() is None:
                    self.proc.kill()
            except Exception:
                pass

    def _worker(self, cmd: str, on_done):
        start = time.time()
        try:
            popen_cmd = cmd if self.use_shell else shlex.split(cmd)
            self.proc = subprocess.Popen(
                popen_cmd,
                shell=self.use_shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                if self._cancel.is_set():
                    break
                self.output_q.put(line)
                if (time.time() - start) > self.timeout:
                    self.output_q.put("\n[!] Timeout reached. Terminating...\n")
                    self.cancel()
                    break
            code = self.proc.wait(timeout=5)
            on_done(code)
        except subprocess.TimeoutExpired:
            self.output_q.put("\n[!] Process hang; killed.\n")
            self.cancel()
            on_done(-1)
        except Exception as e:
            self.output_q.put(f"\n[!] Error: {e}\n")
            on_done(-2)




def log_entry(entry: dict):
    path = user_logs_dir() / f"history_{dt.date.today().isoformat()}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")




class SmartShellGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SmartShell AI")
        self.root.configure(bg="#1e1e1e")
        self.root.geometry("980x620")

        if os.name == "nt":
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                pass

        frame_top = tk.Frame(root, bg="#1e1e1e")
        frame_top.pack(fill=tk.X, padx=10, pady=10)

        self.lbl = tk.Label(frame_top, text="What do you want to do?",
                            bg="#1e1e1e", fg="white", font=("Segoe UI", 11))
        self.lbl.pack(side=tk.LEFT)

        self.entry = tk.Entry(frame_top, width=70, font=("Consolas", 14),
                              bg="#2e2e2e", fg="#fff", insertbackground="#fff")
        self.entry.pack(side=tk.LEFT, padx=10)
        self.entry.bind("<Return>", lambda e: self.on_translate())

        self.btn_run = tk.Button(frame_top, text="⚙️ Translate & Run",
                                 bg="#3a3a3a", fg="white",
                                 command=self.on_translate)
        self.btn_run.pack(side=tk.LEFT, padx=5)

        self.btn_cancel = tk.Button(frame_top, text="❌ Cancel",
                                    bg="#5a2b2b", fg="white",
                                    command=self.on_cancel, state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.LEFT, padx=5)

        self.out = scrolledtext.ScrolledText(root, height=26,
                                             bg="#111", fg="#c9f9c9",
                                             font=("Consolas", 11))
        self.out.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.status = tk.Label(root, text="Ready.", bg="#1e1e1e", fg="#aaa", anchor="w")
        self.status.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.runner = Runner(timeout_sec=SETTINGS.timeout_sec, use_shell=SETTINGS.use_shell)
        self._poll_output()


    def append(self, text: str):
        self.out.insert(tk.END, text)
        self.out.see(tk.END)

    def set_status(self, msg: str):
        self.status.config(text=msg)


    def on_translate(self):
        prompt = self.entry.get().strip()
        if not prompt:
            messagebox.showwarning("SmartShell", "Please enter a prompt.")
            return

        self.append(f"\n💬 Prompt: {prompt}\n")
        self.set_status("Translating…")

        try:
            cmd = translate(prompt)
        except Exception as e:
            self.append(f"❌ Translate failed: {e}\n")
            self.set_status("Translate failed.")
            return

        if not cmd:
            self.append("❌ Empty command.\n")
            self.set_status("Empty command.")
            return


        self.append(f"🧠 Mode: {'LLM' if TRANSLATE_MODE=='llm' else 'Rule-based'}\n")
        self.append(f"💡 Command: {cmd}\n")

        ok = messagebox.askyesno("SmartShell", f"Run this?\n\n{cmd}")
        if not ok:
            self.append("❌ Cancelled by user.\n")
            self.set_status("Cancelled.")
            return

        self.btn_run.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.NORMAL)
        self.set_status("Running… (you can Cancel)")

        start_ts = dt.datetime.now().isoformat(timespec="seconds")

        def on_done(code: int):
            self.btn_run.config(state=tk.NORMAL)
            self.btn_cancel.config(state=tk.DISABLED)
            self.set_status(f"Done (exit {code}).")
            log_entry({
                "time": start_ts,
                "prompt": prompt,
                "command": cmd,
                "exit_code": code,
                "mode": TRANSLATE_MODE,
            })

        self.runner.run_async(cmd, on_done)

    def on_cancel(self):
        self.runner.cancel()
        self.append("\n[⛔] Cancel requested by user.\n")
        self.set_status("Cancelling…")


    def _poll_output(self):
        try:
            while True:
                line = self.runner.output_q.get_nowait()
                self.append(line)
        except queue.Empty:
            pass
        self.root.after(60, self._poll_output)




def main():
    root = tk.Tk()
    app = SmartShellGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
