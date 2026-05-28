# local-llm-inference

Compare inference output and latency between a local Ollama model and a cloud model.

## 1) Setup local LLM with Ollama

1. Install Ollama: https://ollama.com/download
2. Start Ollama service.
3. Pull a local model (example):

```bash
ollama pull llama3.2
```

Optional environment override:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
```

## 2) Setup cloud provider (OpenAI)

Set your API key:

```bash
export OPENAI_API_KEY=<your_key>
```

Optional overrides:

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1
```

## 3) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 4) Run comparison

```bash
python -m src.inference_compare --prompt "Explain local-vs-cloud LLM inference in one paragraph"
```

The command prints:
- local Ollama response + latency
- cloud OpenAI response + latency

## 5) Run tests

```bash
python -m pytest -q
```
