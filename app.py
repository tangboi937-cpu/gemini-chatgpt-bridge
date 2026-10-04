import os
import contextlib
from collections.abc import AsyncIterator

import requests

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


def call_gemini(prompt: str, system_instruction: str = "", thinking_level: str = "medium") -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("Server is missing GEMINI_API_KEY")

    if thinking_level not in ("low", "medium", "high"):
        thinking_level = "medium"

    combined_prompt = prompt

    if system_instruction.strip():
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
        error = data.get("error", {})
        message = error.get(
            "message",
            f"Gemini API request failed with HTTP {response.status_code}"
        )
        raise RuntimeError(message)

    answer = data.get("output_text")

    if isinstance(answer, str) and answer.strip():
        return answer

    return "Gemini returned a response, but no text output was found."


# ---------------------------------------------------------
# MCP SERVER
# ---------------------------------------------------------

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
    Send an explicit prompt to Gemini 3.8 Flash and return its response.
    """

    if not prompt.strip():
        return "Error: prompt is required."

    try:
        return call_gemini(
            prompt=prompt,
            system_instruction=system_instruction,
            thinking_level=thinking_level
        )

    except requests.RequestException as exc:
        return f"Error: Could not reach Gemini API: {exc}"

    except RuntimeError as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

async def health(request):
    return JSONResponse({
        "name": "Gemini 3.8 Flash Bridge",
        "status": "ok",
        "mcp_endpoint": "/mcp"
    })


# ---------------------------------------------------------
# MCP LIFESPAN
# ---------------------------------------------------------

@contextlib.asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    async with mcp.session_manager.run():
        yield


# ---------------------------------------------------------
# RENDER / MCP SECURITY
# ---------------------------------------------------------

security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[
        "gemini-chatgpt-bridge.onrender.com",
        "gemini-chatgpt-bridge.onrender.com:*"
    ],
    allowed_origins=[
        "https://chatgpt.com",
        "https://chat.openai.com"
    ]
)


# ---------------------------------------------------------
# STARLETTE APPLICATION
# ---------------------------------------------------------

app = Starlette(
    routes=[
        Route(
            "/",
            health,
            methods=["GET"]
        ),

        Mount(
            "/",
            app=mcp.streamable_http_app(
                json_response=True,
                stateless_http=True,
                transport_security=security,
                host="gemini-chatgpt-bridge.onrender.com"
            )
        )
    ],

    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=[
                "https://chatgpt.com",
                "https://chat.openai.com"
            ],
            allow_methods=[
                "GET",
                "POST",
                "DELETE",
                "OPTIONS"
            ],
            allow_headers=[
                "Authorization",
                "Content-Type",
                "Last-Event-ID",
                "Mcp-Method",
                "Mcp-Name",
                "Mcp-Protocol-Version",
                "Mcp-Session-Id"
            ],
            expose_headers=[
                "Mcp-Session-Id"
            ]
        )
    ],

    lifespan=lifespan
)
