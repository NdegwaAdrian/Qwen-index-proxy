# Local LLM Compatibility Proxy

A lightweight FastAPI compatibility proxy for connecting OpenAI-compatible clients to a locally hosted LLM through llama.cpp.

## Why I Built This

I wanted to run a quantized language model locally on constrained Intel hardware and connect it to Index 0 using llama.cpp's OpenAI-compatible server.

The model worked directly through llama.cpp, but client integration exposed compatibility problems involving system-message ordering and streaming responses. Instead of modifying llama.cpp or the model, I built a small proxy layer between the client and inference server.

The project became an opportunity to explore local inference, memory constraints, HTTP APIs, asynchronous Python, streaming protocols, debugging, and compatibility between independently developed software components.

## Architecture

```text
Index 0
   |
   | OpenAI-compatible HTTP / SSE
   v
FastAPI Compatibility Proxy
   |
   | HTTP / SSE
   v
llama-server
   |
   v
Qwen3.5 0.8B Q4_0
   |
   v
CPU / Memory
```

The proxy and inference server run as separate processes with separate responsibilities.

The proxy handles protocol and message compatibility, while llama-server performs model inference.

## Current Features

- OpenAI-compatible `/v1/models` forwarding
- OpenAI-compatible `/v1/chat/completions` forwarding
- System-message normalization
- Non-streaming JSON responses
- Server-Sent Events (SSE) streaming
- Health-check endpoint
- Local-only communication between the proxy and inference server

## Engineering Challenges

### 1. System Message Ordering

The client could send multiple system messages at different positions in the conversation. Qwen's chat template rejected this structure because system messages were required at the beginning.

The proxy normalizes the message array by:

1. Collecting system messages.
2. Preserving the order of non-system messages.
3. Combining the system instructions.
4. Inserting a single system message at index `0`.

For example:

```text
Input

system A
user B
assistant C
system D
user E

        |
        v

Normalized

system A + D
user B
assistant C
user E
```

This creates a message structure compatible with the model without requiring changes to the client or llama.cpp.

The transformation traverses the message collection while preserving the relative order of non-system messages.

### 2. Streaming Responses

Non-streaming requests worked through the proxy, but Index 0 requested streaming responses.

The original implementation attempted to parse every upstream response using:

```python
return response.json()
```

This works when llama-server returns one complete JSON document.

A streaming response is different. llama-server returns a sequence of Server-Sent Events rather than one JSON object.

Conceptually:

```text
data: {...}

data: {...}

data: {...}

data: [DONE]
```

Attempting to parse this entire response as one JSON document resulted in a `JSONDecodeError`.

The proxy now detects whether streaming was requested and uses an asynchronous generator to forward response bytes incrementally:

```text
llama-server
      |
      | response chunk
      v
FastAPI proxy
      |
      | yield chunk
      v
Client
```

FastAPI's `StreamingResponse` is used to expose the upstream SSE stream to the client.

### 3. Context Window and Memory Usage

The model metadata advertised a training context of 262,144 tokens.

Allowing llama.cpp to derive the runtime context from the model resulted in a large KV-cache allocation and approximately 3.7 GB of observed process memory usage on the development machine.

That context capacity was unnecessary for the intended local workflow, so the runtime context was explicitly limited to:

```text
4096 tokens
```

Observed process memory subsequently dropped to approximately:

```text
474 MB
```

while retaining enough context for the initial workflow.

This experiment demonstrated that model file size alone does not determine inference memory requirements. Runtime state, particularly the KV cache and selected context size, can contribute significantly to memory consumption.

## Setup

### Prerequisites

This project assumes:

- Python 3.11+
- Conda or another Python environment manager
- A working llama.cpp build containing `llama-server`
- A compatible GGUF model
- An OpenAI-compatible client

The development environment used for this project was:

- macOS 12.7.6
- Intel x86_64
- Python 3.11.13
- llama.cpp built from source using AppleClang
- Qwen3.5 0.8B Q4_0 GGUF
- CPU-based inference

### 1. Clone the Repository

```bash
git clone <repository-url>
cd qwen-index-proxy
```

Replace `<repository-url>` with the repository URL once the project has been published.

### 2. Create the Python Environment

Using Conda:

```bash
conda create -n qwen-proxy python=3.11
conda activate qwen-proxy
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The project's direct Python dependencies are:

- FastAPI
- HTTPX
- Uvicorn

### 3. Start llama-server

Start the local inference server with a 4,096-token runtime context.

For example:

```bash
/path/to/llama-server \
  -m /path/to/model.gguf \
  -c 4096 \
  --port 8080
```

The exact executable and model paths depend on the local llama.cpp installation.

The upstream inference API should then be available at:

```text
http://127.0.0.1:8080
```

### 4. Start the Compatibility Proxy

From the project directory:

```bash
conda activate qwen-proxy

uvicorn proxy:app \
  --host 127.0.0.1 \
  --port 8081
```

For development, automatic reload can optionally be enabled:

```bash
uvicorn proxy:app \
  --host 127.0.0.1 \
  --port 8081 \
  --reload
```

The compatibility proxy should now be available at:

```text
http://127.0.0.1:8081
```

### 5. Verify the Proxy

Check the health endpoint:

```bash
curl http://127.0.0.1:8081/health
```

Expected response:

```json
{"status":"ok"}
```

Check model forwarding:

```bash
curl http://127.0.0.1:8081/v1/models
```

A successful response indicates that the proxy can communicate with llama-server and retrieve its available model information.

### 6. Configure an OpenAI-Compatible Client

Configure the client to use the proxy rather than connecting directly to llama-server.

Use:

```text
Base URL: http://127.0.0.1:8081/v1
```

The resulting communication path is:

```text
Client
   |
   v
127.0.0.1:8081
FastAPI Proxy
   |
   v
127.0.0.1:8080
llama-server
   |
   v
Local GGUF Model
```

## Request Flow

For each chat-completion request:

1. The proxy receives an OpenAI-compatible HTTP request.
2. The JSON request body is decoded.
3. `normalize_messages()` examines the conversation.
4. System messages are separated from normal conversation messages.
5. Multiple system messages are combined when necessary.
6. A single system message is placed at the beginning of the conversation.
7. The normalized request is forwarded to llama-server.
8. The proxy determines whether the client requested a normal or streaming response.
9. Non-streaming responses are returned as JSON.
10. Streaming responses are forwarded incrementally using SSE.

The proxy does not perform model inference itself. Its responsibility is compatibility and communication between the client and the local inference server.

## Performance Observations

The project was developed on a 2017 Intel-based MacBook Air with 8 GB of RAM.

During experimentation, allowing llama.cpp to derive the context size from the model metadata resulted in approximately 3.7 GB of observed process memory usage.

Explicitly limiting the runtime context to 4,096 tokens reduced observed process memory to approximately 474 MB.

Example generation throughput observed during testing was approximately:

```text
10–13 tokens/second
```

depending on the request.

Prompt-processing speed and generation speed varied according to the workload. Larger client prompts also require additional processing before generation begins.

These measurements are specific to the development machine, model, configuration, and workload and should not be interpreted as general Qwen or llama.cpp benchmarks.

## Project Structure

```text
qwen-index-proxy/
├── .gitignore
├── proxy.py
├── README.md
└── requirements.txt
```

### `proxy.py`

Contains the FastAPI application, health endpoint, model forwarding, message-normalization logic, chat-completion forwarding, and SSE streaming implementation.

### `requirements.txt`

Defines the direct Python dependencies required to run the proxy.

### `.gitignore`

Prevents generated Python files, local environments, environment-variable files, logs, editor-specific files, and other local artifacts from entering version control.

## Current Limitations

This project is currently a lightweight compatibility layer rather than a production API gateway.

Current limitations include:

- No authentication or authorization layer
- No rate limiting
- No automated test suite yet
- Limited upstream error handling
- No structured application logging
- Server configuration is currently defined directly in the source code
- Performance is constrained by CPU-only inference on the development machine
- The configured 4,096-token runtime context may be insufficient for larger workloads
- Compatibility has currently been tested against a limited model/client configuration

## Roadmap

Potential next steps include:

- [ ] Add unit tests for message normalization
- [ ] Test zero, one, and multiple system-message cases
- [ ] Add integration tests for streaming and non-streaming requests
- [ ] Improve upstream HTTP error propagation
- [ ] Add graceful handling when llama-server is unavailable
- [ ] Replace temporary console debugging with structured logging
- [ ] Move server configuration to environment variables
- [ ] Benchmark different context-window sizes
- [ ] Measure time to first token
- [ ] Measure prompt-processing throughput
- [ ] Measure generation throughput
- [ ] Test additional GGUF models and quantization levels
- [ ] Document compatibility behavior across different clients

## What I Learned

Building this project provided practical experience across several areas of computer science and software engineering.

### Systems and Computer Architecture

- Compiling native C/C++ software using CMake
- Working with an x86_64 environment
- Understanding CPU-based model inference
- Observing process memory and CPU utilization
- Understanding the relationship between model weights, context size, and runtime memory
- Investigating KV-cache allocation

### Backend and Network Programming

- HTTP client-server communication
- Localhost networking and ports
- REST-style API endpoints
- JSON request and response handling
- FastAPI
- Uvicorn
- HTTPX
- Server-Sent Events
- Streaming HTTP responses

### Python

- Functions
- Lists and dictionaries
- Control flow
- Message transformation
- `async` / `await`
- Asynchronous iteration
- Generators and `yield`
- Dependency isolation with Conda

### Debugging and Testing

The development process involved diagnosing several different classes of failure, including:

- HTTP `404 Not Found`
- HTTP `500 Internal Server Error`
- Missing Python imports
- `UnboundLocalError`
- JSON parsing failures
- Chat-template incompatibilities
- Branch-specific behavior
- Streaming versus non-streaming responses

Reading stack traces and isolating individual layers of the system was an important part of the development process.

### Software Engineering

- Git repository initialization
- Staging and committing changes
- `.gitignore`
- Dependency management
- Incremental development
- Separation of concerns
- Designing around compatibility boundaries
- Documenting technical decisions and limitations

## Status

The core local inference pipeline is operational:

```text
OpenAI-Compatible Client
        |
        v
FastAPI Compatibility Proxy
        |
        v
llama-server
        |
        v
Quantized Local LLM
```

Both normal JSON completions and streamed SSE completions are supported.

The next development phase focuses on automated testing, error handling, configuration management, and performance measurement.

## License

No license has been selected yet.