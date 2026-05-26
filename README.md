# local-llm-inference

Run a local LLM using [Ollama](https://ollama.com/) for offline inference.

## Prerequisites

- macOS, Linux, or Windows (WSL2 recommended on Windows)
- At least 8 GB RAM (16 GB+ recommended for larger models)
- Disk space for model weights (typically 4 GB+)

## 1) Install Ollama

### macOS / Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows

Download and install from: https://ollama.com/download

## 2) Start Ollama

On macOS and Windows, Ollama usually starts automatically after installation.

On Linux, start the service:

```bash
ollama serve
```

Keep this terminal running.

## 3) Pull and run a model

Example with Llama 3:

```bash
ollama run llama3
```

The first run downloads the model, then opens an interactive chat in your terminal.

## 4) Use Ollama HTTP API (optional)

In another terminal:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt": "Explain local LLM inference in one paragraph.",
  "stream": false
}'
```

## Useful commands

```bash
ollama list              # list downloaded models
ollama pull mistral      # download another model
ollama run mistral       # run a different model
ollama rm llama3         # remove a model
```