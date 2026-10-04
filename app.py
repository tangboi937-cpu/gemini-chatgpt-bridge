        import os
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
