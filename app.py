import os
import contextlib
import asyncio
from collections.abc import AsyncIterator

from google import genai

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is missing")


# Google Gemini client
client = genai.Client(
    api_key=GEMINI_API_KEY
) if GEMINI_API_KEY else None


async def call_gemini(
    prompt: str,
    system_instruction: str = "",
    thinking_level: str = "medium"
) -> str:

    if client is None:
        return "Error: GEMINI_API_KEY is missing on Render."

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

    def make_request():
        return client.interactions.create(
            model="gemini-3.8-flash",
            input=combined_prompt,
            generation_config={
                "thinking_level": thinking_level
            }
        )

    try:
        interaction = await asyncio.to_thread(make_request)

        answer = interaction.output_text

        if answer:
            return answer

        return "Gemini returned no text."

    except Exception as exc:
        print(
            f"Gemini request failed: "
            f"{type(exc).__name__}: {exc}"
        )

        return (
            f"Gemini API error: "
            f"{type(exc).__name__}: {exc}"
        )


# ---------------------------------------------------------
# MCP SERVER
# ---------------------------------------------------------

mcp = MCPServer(
    "Gemini 3.8 Flash Bridge"
)


@mcp.tool()
async def send_prompt(
    prompt: str,
    system_instruction: str = "",
    thinking_level: str = "medium"
) -> str:
    """
    Send an explicit prompt to Gemini 3.8 Flash.
    """

    if not prompt or not prompt.strip():
        return "Error: prompt is required."

    return await call_gemini(
        prompt=prompt,
        system_instruction=system_instruction,
        thinking_level=thinking_level
    )


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
# MCP SECURITY
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
# STARLETTE
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
            allow_headers=["*"],
            expose_headers=[
                "Mcp-Session-Id"
            ]
        )
    ],

    lifespan=lifespan
)
