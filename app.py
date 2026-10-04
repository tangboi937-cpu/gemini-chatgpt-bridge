import os
import requests
import contextlib

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount

from mcp.server.mcpserver import MCPServer


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PORT = int(os.environ.get("PORT", "10000"))


def call_gemini(prompt, system_instruction=None, thinking_level="medium"):
    if not GEMINI_API_KEY:
        raise RuntimeError("Server is missing GEMINI_API_KEY")

    if thinking_level not in ("low", "medium", "high"):
        thinking_level = "medium"

    combined_prompt = prompt

    if system_instruction:
        combined_prompt = (
            f"System instructions:\n{system_instruction}\n\n"
            f"User:\n{prompt}"
        )

    payload = {
        "model": "gemini-3.8-flash",
        "input": combined_prompt,
        "generation_config": {
            "thinking_level": thinking_level
        }
    }

    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        headers={
            "x-goog-api-key": GEMINI_API_KEY,
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=120
    )

    try:
        data = response.json()
    except ValueError:
        data = {}

    if not response.ok:
        message = data.get("error", {}).get(
            "message",
            "Gemini API request failed"
        )
        raise RuntimeError(message)

    answer = data.get("output_text")

    if isinstance(answer, str) and answer.strip():
        return answer

    return "Gemini returned a response, but no text output was found."


mcp = MCPServer(
    "Gemini 3.8 Flash Bridge"
)


@mcp.tool()
def send_prompt(
    prompt: str,
    system_instruction: str = "",
    thinking_level: str = "medium"
) -> str:
    """
    Send an explicit user-provided prompt to Gemini 3.8 Flash
    and return Gemini's response.
    """

    if not prompt.strip():
        return "Error: prompt is required."

    try:
        return call_gemini(
            prompt,
            system_instruction=system_instruction or None,
            thinking_level=thinking_level
        )
    except requests.RequestException:
        return "Error: Could not reach Gemini API."
    except RuntimeError as exc:
        return f"Error: {exc}"


async def health(request):
    return JSONResponse({
        "name": "Gemini 3.8 Flash ChatGPT Action Bridge",
        "status": "ok",
        "endpoint": "/mcp",
        "mcp_status": "enabled"
    })


@contextlib.asynccontextmanager
async def lifespan(app):
    async with mcp.session_manager.run():
        yield


app = Starlette(
    routes=[
        Route("/", health, methods=["GET"]),
        Mount(
            "/",
            app=mcp.streamable_http_app(
                json_response=True,
                stateless_http=True,
                host="0.0.0.0"
            )
        )
    ],
    lifespan=lifespan
)
