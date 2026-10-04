        import os
import requests
from flask import Flask, request, jsonify, make_response

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
BRIDGE_API_KEY = os.environ.get("BRIDGE_API_KEY")
PORT = int(os.environ.get("PORT", "3000"))

MCP_PROTOCOL_VERSION = "2025-06-18"
MCP_SERVER_NAME = "gemini-3-8-flash-bridge"
MCP_SERVER_VERSION = "0.3.0"


def cors_response(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Accept, Authorization, Mcp-Session-Id, "
        "MCP-Protocol-Version, X-Bridge-Key"
    )
    response.headers["Access-Control-Expose-Headers"] = "Mcp-Session-Id"
    return response


def jsonrpc_result(request_id, result):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }


def jsonrpc_error(request_id, code, message):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


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
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
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

    parts = []

    def collect_text(value):
        if isinstance(value, dict):
            text = value.get("text")

            if isinstance(text, str) and text.strip():
                parts.append(text)

            for child in value.values():
                collect_text(child)

        elif isinstance(value, list):
            for child in value:
                collect_text(child)

    collect_text(data)

    if parts:
        return "\n".join(parts)

    return "Gemini returned a response, but no text output was found."


@app.get("/")
def health():
    return jsonify({
        "name": "Gemini 3.8 Flash ChatGPT Action Bridge",
        "status": "ok",
        "endpoint": "/gemini",
        "mcp_endpoint": "/mcp",
        "mcp_status": "enabled",
    })


@app.post("/gemini")
def gemini():
    if request.headers.get("X-Bridge-Key") != BRIDGE_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    body = request.get_json(silent=True) or {}

    prompt = body.get("prompt")
    system_instruction = body.get("system_instruction")
    thinking_level = body.get("thinking_level", "medium")

    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({"error": "prompt is required"}), 400

    try:
        answer = call_gemini(
            prompt,
            system_instruction=system_instruction,
            thinking_level=thinking_level,
        )

        return jsonify({
            "answer": answer,
            "model": "gemini-3.8-flash",
        })

    except requests.RequestException:
        return jsonify({
            "error": "Could not reach Gemini API"
        }), 502

    except RuntimeError as exc:
        return jsonify({
            "error": str(exc)
        }), 500


@app.route("/mcp", methods=["OPTIONS"])
def mcp_options():
    return cors_response(make_response("", 204))


@app.route("/mcp", methods=["GET", "DELETE"])
def mcp_non_post():
    response = make_response(
        jsonify({
            "error": "This MCP endpoint uses POST with JSON responses."
        }),
        405,
    )

    response.headers["Allow"] = "POST, OPTIONS"

    return cors_response(response)


@app.post("/mcp")
def mcp():
    body = request.get_json(silent=True)

    if not isinstance(body, dict):
        response = jsonify(
            jsonrpc_error(
                None,
                -32600,
                "Invalid JSON-RPC request"
            )
        )

        response.status_code = 400

        return cors_response(response)

    method = body.get("method")
    request_id = body.get("id")

    if method == "notifications/initialized":
        return cors_response(make_response("", 202))

    if method == "initialize":
        params = body.get("params") or {}

        requested_version = params.get(
            "protocolVersion",
            MCP_PROTOCOL_VERSION
        )

        negotiated = (
            requested_version
            if requested_version in (
                "2025-06-18",
                "2025-03-26"
            )
            else MCP_PROTOCOL_VERSION
        )

        result = {
            "protocolVersion": negotiated,
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": MCP_SERVER_NAME,
                "version": MCP_SERVER_VERSION,
            },
            "instructions": (
                "Use send_prompt when the user explicitly asks to send "
                "a prompt to Gemini 3.8 Flash."
            ),
        }

        response = jsonify(
            jsonrpc_result(
                request_id,
                result
            )
        )

        response.headers["MCP-Protocol-Version"] = negotiated

        return cors_response(response)

    if method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "send_prompt",
                    "description": (
                        "Send an explicit user-provided prompt to Gemini "
                        "3.8 Flash and return Gemini's response."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": (
                                    "The exact prompt the user wants "
                                    "sent to Gemini."
                                ),
                            },
                            "system_instruction": {
                                "type": "string",
                                "description": (
                                    "Optional behavior instructions "
                                    "for Gemini."
                                ),
                            },
                            "thinking_level": {
                                "type": "string",
                                "enum": [
                                    "low",
                                    "medium",
                                    "high"
                                ],
                                "default": "medium",
                            },
                        },
                        "required": ["prompt"],
                        "additionalProperties": False,
                    },
                }
            ]
        }

        response = jsonify(
            jsonrpc_result(
                request_id,
                result
            )
        )

        return cors_response(response)

    if method == "tools/call":
        params = body.get("params") or {}

        tool_name = params.get("name")
        arguments = params.get("arguments") or {}

        if tool_name != "send_prompt":
            response = jsonify(
                jsonrpc_error(
                    request_id,
                    -32602,
                    "Unknown tool"
                )
            )

            response.status_code = 400

            return cors_response(response)

        prompt = arguments.get("prompt")

        if not isinstance(prompt, str) or not prompt.strip():
            response = jsonify(
                jsonrpc_error(
                    request_id,
                    -32602,
                    "prompt is required"
                )
            )

            response.status_code = 400

            return cors_response(response)

        try:
            answer = call_gemini(
                prompt,
                system_instruction=arguments.get(
                    "system_instruction"
                ),
                thinking_level=arguments.get(
                    "thinking_level",
                    "medium"
                ),
            )

            result = {
                "content": [
                    {
                        "type": "text",
                        "text": answer,
                    }
                ]
            }

            response = jsonify(
                jsonrpc_result(
                    request_id,
                    result
                )
            )

            return cors_response(response)

        except requests.RequestException:
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": "Could not reach Gemini API.",
                    }
                ],
                "isError": True,
            }

            response = jsonify(
                jsonrpc_result(
                    request_id,
                    result
                )
            )

            return cors_response(response)

        except RuntimeError as exc:
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": str(exc),
                    }
                ],
                "isError": True,
            }

            response = jsonify(
                jsonrpc_result(
                    request_id,
                    result
                )
            )

            return cors_response(response)

    response = jsonify(
        jsonrpc_error(
            request_id,
            -32601,
            "Method not found"
        )
    )

    response.status_code = 404

    return cors_response(response)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=PORT
                            )
