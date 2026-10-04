import os
import contextlib
from collections.abc import AsyncIterator

import httpx

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not configured")


async def call_gemini(
    prompt: str,
    system_instruction: str = "",
    thinking_level: str = "medium"
) -> str:

    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY is missing on the server."

    if thinking_level not in ("low", "medium", "high"):
        thinking_level = "medium"

    combined_prompt = prompt.strip()

    if system_instruction.strip():
        combined_prompt = (
            "System instructions:\n"
            + system_instruction.strip()
            + "\n\nUser:\n"
            + prompt.strip()
        )

    payload = {
        "model": "gemini-3.8-flash",
        "input": combined_prompt,
        "generation_config": {
            "thinking_level": thinking_level
        }
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:

            response = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/interactions",
                headers={
                    "x-goog-api-key": GEMINI_API_KEY,
                    "Content-Type": "application/json"
                },
                json=payload
            )

        try:
            data = response.json()
        except Exception:
            data = {}

        if response.status_code < 200 or response.status_code >= 300:

            error = data.get("error", {})

            message = error.get("message")

            if not message:
                message = response.text[:1000]

            return (
                f"Gemini API error "
                f"(HTTP {response.status_code}): {message}"
            )

        answer = data.get("output_text")

        if isinstance(answer, str) and answer.strip():
            return answer.strip()

        # Some API responses may put the text in output.
        output = data.get("output")

        if isinstance(output, list):
            parts = []

            for item in output:
                if isinstance(item, dict):
                    text = item.get("text")

                    if isinstance(text, str):
                        parts.append(text)

            if parts:
                return "\n".join(parts).strip()

        return (
            "Gemini returned successfully, but no text output was found. "
            f"Raw response: {str(data)[:2000]}"
        )

    except httpx.TimeoutException:
        return "Error: Gemini API request timed out."

    except httpx.RequestError as exc:
        return f"Error connecting to Gemini API: {exc}"

    except Exception as exc:
        return f"Unexpected Gemini bridge error: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------
# MCP
# ---------------------------------------------------------

mcp = MCPServer("Gemini 3.8 Flash Bridge")


@mcp.tool()
async def send_prompt(
    prompt: str,
    system_instruction: str = "",
    thinking_level: str = "medium"
) -> str:
    """
    Send a prompt to Gemini 3.8 Flash.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        return "Error: prompt is required."

    return await call_gemini(
        prompt=prompt,
        system_instruction=system_instruction,
        thinking_level=thinking_level
    )


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

async def health(request):
    return JSONResponse({
        "name": "Gemini 3.8 Flash Bridge",
        "status": "ok",
        "mcp_endpoint": "/mcp"
    })


# ---------------------------------------------------------
# Lifespan
# ---------------------------------------------------------

@contextlib.asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    async with mcp.session_manager.run():
        yield


# ---------------------------------------------------------
# MCP transport security
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
# Application
# ---------------------------------------------------------

app = Starlette(
    routes=[
        Route("/", health, methods=["GET"]),

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
            allow_headers=["*"],
            expose_headers=[
                "Mcp-Session-Id"
            ]
        )
    ],

    lifespan=lifespan
)
