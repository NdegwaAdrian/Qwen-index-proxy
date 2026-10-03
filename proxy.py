from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx

app = FastAPI()

LLAMA_SERVER = "http://127.0.0.1:8080"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{LLAMA_SERVER}/v1/models")

    return response.json()

#normalize messages algo to separate system messages from normal messages
def normalize_messages(messages):
    system_messages = []
    normal_messages = []

    for message in messages:
        if message["role"] == "system":
            system_messages.append(message["content"])
        else:
            normal_messages.append(message)

    if system_messages:
        combined_system = "\n\n".join(system_messages)

        normal_messages.insert(
            0,
            {
                "role": "system",
                "content": combined_system
            }
        )

    return normal_messages




async def stream_from_llama(body):
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{LLAMA_SERVER}/v1/chat/completions",
            json=body
        ) as response:
            
            async for chunk in response.aiter_bytes():
                    yield chunk




@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    print("STREAM REQUESTED:", body.get("stream", False))

    body["messages"] = normalize_messages(body["messages"])

    if body.get("stream", False):
        return StreamingResponse(
            stream_from_llama(body), media_type="text/event-stream")



    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{LLAMA_SERVER}/v1/chat/completions",
            json=body
        )

    return response.json()
