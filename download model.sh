ls -lh ./models ~/.config/SmartShellAI/models 2>/dev/null
rm -f ./models/wizardcoder-python-7b-v1.0.Q4_K_M.gguf
rm -f ~/.config/SmartShellAI/models/wizardcoder-python-7b-v1.0.Q4_K_M.gguf
sudo apt install -y aria2
mkdir -p ./models
aria2c -x16 -s16 -k1M \
  "https://huggingface.co/TheBloke/WizardCoder-Python-7B-V1.0-GGUF/resolve/main/wizardcoder-python-7b-v1.0.Q4_K_M.gguf" \
  -d ./models -o wizardcoder-python-7b-v1.0.Q4_K_M.gguf
