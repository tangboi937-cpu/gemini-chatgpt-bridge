        return jsonify({"error": str(exc)}), 500


@app.route("/mcp", methods=["OPTIONS"])
def mcp_options():
    return cors_response(make_response("", 204))


@app.route("/mcp", methods=["GET", "DELETE"])
def mcp_non_post():
    # This plugin uses stateless JSON responses. No persistent MCP session
    # is required, so GET/DELETE are intentionally not used for tool calls.
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
    """ Stateless MCP Streamable HTTP endpoint. It implements the handshake-era MCP methods needed by this plugin: - initialize - notifications/initialized - tools/list - tools/call JSON responses are used instead of an SSE stream, which is supported by OpenAI's Streamable HTTP quickstart for stateless MCP servers. """
    body = request.get_json(silent=True)

    if not isinstance(body, dict):
        response = jsonify(jsonrpc_error(None, -32600, "Invalid JSON-RPC request"))
        response.status_code = 400
        return cors_response(response)

    method = body.get("method")
    request_id = body.get("id")

    # MCP notifications do not receive a JSON-RPC response.
    if method == "notifications/initialized":
        return cors_response(make_response("", 202))

    if method == "initialize":
        params = body.get("params") or {}
        requested_version = params.get("protocolVersion", MCP_PROTOCOL_VERSION)

        # Negotiate the protocol version. This server intentionally supports
        # the 2025-06-18 handshake used by the plugin.
        negotiated = (
            requested_version
            if requested_version in ("2025-06-18", "2025-03-26")
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
                "Use send_prompt when the user explicitly asks to send a "
                "prompt to Gemini 3.8 Flash."
            ),
        }

        response = jsonify(jsonrpc_result(request_id, result))
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
                                    "The exact prompt the user wants sent to Gemini."
                                ),
                            },
                            "system_instruction": {
                                "type": "string",
                                "description": (
                                    "Optional behavior instructions for Gemini."
                                ),
                            },
                            "thinking_level": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                                "default": "medium",
                            },
                        },
                        "required": ["prompt"],
                        "additionalProperties": False,
                    },
                }
            ]
        }

        response = jsonify(jsonrpc_result(request_id, result))
        return cors_response(response)

    if method == "tools/call":
        params = body.get("params") or {}
        tool_name = params.get("name")
        arguments = params.get("arguments") or {}

        if tool_name != "send_prompt":
            response = jsonify(
                jsonrpc_error(request_id, -32602, "Unknown tool")
            )
            response.status_code = 400
            return cors_response(response)

        prompt = arguments.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            response = jsonify(
                jsonrpc_error(request_id, -32602, "prompt is required")
            )
            response.status_code = 400
            return cors_response(response)

        try:
            answer = call_gemini(
                prompt,
                system_instruction=arguments.get("system_instruction"),
                thinking_level=arguments.get("thinking_level", "medium"),
            )

            result = {
                "content": [
                    {
                        "type": "text",
                        "text": answer,
                    }
                ]
            }
            response = jsonify(jsonrpc_result(request_id, result))
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
            response = jsonify(jsonrpc_result(request_id, result))
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
            response = jsonify(jsonrpc_result(request_id, result))
            return cors_response(response)

    response = jsonify(
        jsonrpc_error(request_id, -32601, "Method not found")
    )
    response.status_code = 404
    return cors_response(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
