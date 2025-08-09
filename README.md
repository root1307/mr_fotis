 mr_fotis

SmartShell AI is an intelligent terminal assistant that understands natural language (Greek/English) and translates it into bash or PowerShell commands. It confirms execution before running commands, keeps a log of actions with timestamps, supports cancellation of running tasks, and runs with a local LLM (LlamaC++ + GGUF models). If the required model is not present, SmartShell AI will automatically download it on first run; after that it operates fully offline. This repository contains the Python program and packaging instructions for both Linux and Windows.

## Features
- Translates natural language instructions (Greek/English) into a single-line command for the current OS (bash or PowerShell).
- Requests confirmation before executing the command.
- Keeps a JSONL log of all actions (prompt, command, exit code, timestamp).
- Supports cancellation of running commands and sets a configurable timeout.
- Automatically installs missing Python dependencies (`llama_cpp`, `diskcache`) and downloads the required GGUF model on first run.
- Works offline after the initial download.
- Voice command and hands‑free operation support (coming soon).

## Running from source on Linux or Windows
0. download the model run the.sh file!!!!
```bash
chmod+x download model.sh
./download model.sh
```
1. Clone this repository:
   ```bash
   git clone https://github.com/root1307/mr_fotis.git
   cd mr_fotis
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # on Linux/Mac
   venv\Scripts\activate      # on Windows
   ```
3. Install the dependencies via pip:
   ```bash
   pip install -r requirements.txt
   ```
4. On the first run, the program will automatically download the `wizardcoder-python-7b-v1.0.Q4_K_M.gguf` model if it does not exist (internet connection required). You can also manually place it into a folder named `models` in the project root to avoid downloading.
5. Run the program:
   ```bash
   python smartshell.py
   ```
   The script will automatically install missing Python dependencies (`llama_cpp`, `diskcache`) if they are not already installed.

## Files
- `smartshell.py` – main Python script with GUI and translation logic.
- `requirements.txt` – Python dependencies needed to run from source (`llama_cpp>=0.2.20`).
- `models/` – directory where you can place a `.gguf` model file (optional, the program will download the model on first run if missing).
