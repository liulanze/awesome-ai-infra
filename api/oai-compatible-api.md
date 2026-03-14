# OpenAI-Compatible API

OpenAI-Compatible = A server that uses the same request/response format as OpenAI's API, so you can use OpenAI's client libraries and tools to talk to it.

## Separation of Layers

```
┌─────────────────────────────────────┐
│  API Format (OpenAI-compatible)     │ ← How you communicate
├─────────────────────────────────────┤
│  Serving Framework (vLLM, SGLang)   │ ← Handles requests, batching, optimization
├─────────────────────────────────────┤
│  Model (Qwen, Llama, Mistral, etc.) │ ← The actual neural network
└─────────────────────────────────────┘
```

## Request Flow

1. Your code (using OpenAI SDK):

```python
client.chat.completions.create(
    model="Qwen",
    messages=[{"role": "user", "content": "Hello"}],
)
```

2. OpenAI library sends HTTP POST request:
   `POST http://localhost:8000/v1/chat/completions`
   Body: `{"model": "Qwen", "messages": [...]}`

3. vLLM receives request → parses OpenAI-format JSON → extracts prompt, parameters

4. vLLM runs inference → loads Qwen model → uses continuous batching, paged attention → generates tokens

5. vLLM formats response (OpenAI format):
   `{"choices": [{"message": {"content": "..."}}]}`

6. Your code receives response:
   `response.choices[0].message.content`

## Common Parameters

```json
{
  "temperature": 0.7,
  "max_tokens": 100,
  "top_p": 0.9,
  "frequency_penalty": 0.5,
  "stop": ["\n\n"],
  "stream": true
}
```

## Message Format

```json
{
  "messages": [
    {"role": "system", "content": "You are a Python expert"},
    {"role": "user", "content": "What is a list?"},
    {"role": "assistant", "content": "A list is..."},
    {"role": "user", "content": "Give me an example"}
  ]
}
```

## Response Format

```json
{
  "choices": [...],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 42,
    "total_tokens": 57
  },
  "finish_reason": "stop"
}
```
