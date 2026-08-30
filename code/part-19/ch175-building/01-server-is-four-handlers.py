# -*- coding: utf-8 -*-
# Extracted from: Chapter 175 — Building an MCP Server and Client
# Source: src/.../ch175-building.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A working MCP server, written against the wire format rather than a library.

Everything here follows cite:mcp2026spec's 2026-07-28 revision:

  JSON-RPC 2.0, UTF-8, newline-delimited over stdio
  stateless, self-contained requests
  the protocol version travels per request in
      _meta["io.modelcontextprotocol/protocolVersion"]
  server/discover is MANDATORY to implement and optional to call
  an unsupported version returns UnsupportedProtocolVersionError LISTING the
      versions the server does support, so the caller can retry into the overlap
  servers never initiate requests; clients never send responses

Writing it against the wire is the point. A library hides exactly the three
things this part measured -- version negotiation, statelessness, and what a tool
description carries (eq:server-is-four-handlers).
"""
import hashlib
import io
import json

# A server supports a WINDOW of revisions, which ch:mcp-why measured as the
# dominant lever on ecosystem connectivity. Newest first.
SUPPORTED = ["2026-07-28", "2025-11-25", "2025-06-18"]
VERSION_KEY = "io.modelcontextprotocol/protocolVersion"

# JSON-RPC error codes: -32700..-32600 are reserved by JSON-RPC itself, and
# implementations use the -32000 block for their own.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
UNSUPPORTED_PROTOCOL_VERSION = -32000

# --- the capability this server actually exposes -----------------------------
# ch:mcp-schemas: a description earns its tokens by DISTINGUISHING, not by
# explaining. Every sentence below is there to prevent a wrong selection.

TOOLS = [
    {
        "name": "convert_currency",
        "description": ("Convert an amount between two currency codes at "
                        "today's rate. Not for historical dates."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "frm": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
                "to": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
            },
            "required": ["amount", "frm", "to"],
        },
        # ch:as-state-machines' audit, published rather than kept in a wiki.
        "annotations": {"readOnlyHint": True, "idempotentHint": True},
    },
    {
        "name": "record_expense",
        "description": ("Append an expense to the ledger. Writes; not "
                        "reversible. Use convert_currency first if needed."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
                "memo": {"type": "string"},
                # The deduplication key ch:as-state-machines requires: derived
                # from the run and step, not generated here.
                "idempotency_key": {"type": "string"},
            },
            "required": ["amount", "currency", "idempotency_key"],
        },
        "annotations": {"readOnlyHint": False, "idempotentHint": True},
    },
]

RESOURCES = [
    {"uri": "ledger://policy/limits",
     "name": "expense limits",
     "mimeType": "text/plain"},
]

RESOURCE_BODIES = {
    "ledger://policy/limits": "single expense cap: 500 USD\nrequires memo: yes",
}

RATES = {("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("EUR", "USD"): 1.09,
         ("EUR", "GBP"): 0.86, ("GBP", "USD"): 1.27, ("GBP", "EUR"): 1.16}

LEDGER = []
SEEN_KEYS = {}


def error(req_id, code, message, data=None):
    e = {"code": code, "message": message}
    if data is not None:
        e["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": e}


def result(req_id, payload):
    return {"jsonrpc": "2.0", "id": req_id, "result": payload}


def negotiate(req):
    """Per-request version check. Returns the agreed version or an error dict."""
    meta = (req.get("params") or {}).get("_meta") or {}
    asked = meta.get(VERSION_KEY)
    if asked is None:
        # Absent means the caller is pre-2026 or optimistic; serve the oldest
        # we support, which is the conservative reading.
        return SUPPORTED[-1], None
    if asked not in SUPPORTED:
        return None, error(
            req.get("id"), UNSUPPORTED_PROTOCOL_VERSION,
            "UnsupportedProtocolVersionError",
            # The listing is the whole point: it lets the caller retry into
            # the overlap instead of merely failing (ch:mcp-why).
            {"supported": SUPPORTED, "requested": asked})
    return asked, None


def h_discover(_params, version):
    """MANDATORY to implement. Calling it is optional -- a client may send any
    request directly and handle a version error if one comes back."""
    return {
        "protocolVersions": SUPPORTED,
        "serverInfo": {"name": "ledger", "version": "1.4.0"},
        "capabilities": {"tools": {"listChanged": False},
                         "resources": {"subscribe": False}},
    }


def h_tools_list(_params, version):
    return {"tools": TOOLS}


def h_resources_list(_params, version):
    return {"resources": RESOURCES}


def h_resources_read(params, version):
    uri = params.get("uri")
    if uri not in RESOURCE_BODIES:
        return {"__error__": (INVALID_PARAMS, "unknown resource uri: %r" % uri)}
    return {"contents": [{"uri": uri, "mimeType": "text/plain",
                          "text": RESOURCE_BODIES[uri]}]}


def h_tools_call(params, version):
    name = params.get("name")
    args = params.get("arguments") or {}
    spec = next((t for t in TOOLS if t["name"] == name), None)
    if spec is None:
        # ch:ag-tool-calling: the error message is the cheapest selector in the
        # book. Name what was wrong and enumerate the valid choices.
        return {"__error__": (INVALID_PARAMS,
                              "no tool named %r; available: %s"
                              % (name, ", ".join(t["name"] for t in TOOLS)))}
    missing = [k for k in spec["inputSchema"]["required"] if k not in args]
    if missing:
        return {"__error__": (INVALID_PARAMS,
                              "missing required argument(s) %s for %s"
                              % (", ".join(missing), name))}
    for key, prop in spec["inputSchema"]["properties"].items():
        if key in args and "enum" in prop and args[key] not in prop["enum"]:
            return {"__error__": (INVALID_PARAMS,
                                  "%s=%r is not valid for %s; expected one of %s"
                                  % (key, args[key], name,
                                     ", ".join(prop["enum"])))}
    if name == "convert_currency":
        frm, to = args["frm"], args["to"]
        rate = 1.0 if frm == to else RATES.get((frm, to))
        if rate is None:
            return {"__error__": (INVALID_PARAMS,
                                  "no rate for %s->%s" % (frm, to))}
        out = round(args["amount"] * rate, 2)
        # Echo the identity of what was answered, so a mis-selection is
        # detectable rather than silent (ch:mcp-primitives).
        return {"content": [{"type": "text",
                             "text": "%.2f %s = %.2f %s" % (args["amount"], frm,
                                                            out, to)}],
                "structuredContent": {"amount": out, "currency": to,
                                      "rate": rate}}
    if name == "record_expense":
        key = args["idempotency_key"]
        # ch:as-state-machines: the key is written BEFORE the effect, so a
        # replay after a crash between the two finds it present.
        if key in SEEN_KEYS:
            return {"content": [{"type": "text", "text": "already recorded"}],
                    "structuredContent": {"entry": SEEN_KEYS[key],
                                          "duplicate": True}}
        entry = len(LEDGER) + 1
        SEEN_KEYS[key] = entry
        LEDGER.append({"entry": entry, "amount": args["amount"],
                       "currency": args["currency"],
                       "memo": args.get("memo", "")})
        return {"content": [{"type": "text", "text": "recorded entry %d" % entry}],
                "structuredContent": {"entry": entry, "duplicate": False}}
    return {"__error__": (METHOD_NOT_FOUND, "unhandled tool %r" % name)}


HANDLERS = {
    "server/discover": h_discover,
    "tools/list": h_tools_list,
    "tools/call": h_tools_call,
    "resources/list": h_resources_list,
    "resources/read": h_resources_read,
}


def handle(req):
    """One self-contained request in, one response out. No session state."""
    if req.get("jsonrpc") != "2.0" or "method" not in req:
        return error(req.get("id"), INVALID_REQUEST, "not a JSON-RPC 2.0 request")
    version, err = negotiate(req)
    if err is not None:
        return err
    fn = HANDLERS.get(req["method"])
    if fn is None:
        return error(req.get("id"), METHOD_NOT_FOUND,
                     "unknown method %r; available: %s"
                     % (req["method"], ", ".join(sorted(HANDLERS))))
    out = fn(req.get("params") or {}, version)
    if isinstance(out, dict) and "__error__" in out:
        code, msg = out["__error__"]
        return error(req.get("id"), code, msg)
    # A notification -- no id -- gets no response, per JSON-RPC.
    if "id" not in req:
        return None
    return result(req["id"], out)


def serve_line(line):
    """The stdio binding: newline-delimited JSON-RPC over a byte stream."""
    try:
        req = json.loads(line)
    except ValueError:
        return json.dumps(error(None, PARSE_ERROR, "invalid JSON"))
    resp = handle(req)
    return None if resp is None else json.dumps(resp)


def call(method, params=None, version=SUPPORTED[0], req_id=1):
    """Test helper: build a self-contained request and run it through the wire."""
    p = dict(params or {})
    p["_meta"] = {VERSION_KEY: version}
    line = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method,
                       "params": p})
    out = serve_line(line)
    return json.loads(out) if out else None


def tool_digest():
    """ch:mcp-security: hash the definitions at approval and compare on every
    listing, because approval is a snapshot of a mutable thing."""
    blob = json.dumps(TOOLS, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


print("A working MCP server over the 2026-07-28 wire format.")
print("Supported revisions: %s" % ", ".join(SUPPORTED))
print()

print("1. server/discover -- mandatory to implement, optional to call")
d = call("server/discover")["result"]
print("   versions:     %s" % ", ".join(d["protocolVersions"]))
print("   serverInfo:   %s %s" % (d["serverInfo"]["name"],
                                  d["serverInfo"]["version"]))
print("   capabilities: %s" % ", ".join(sorted(d["capabilities"])))
print()

print("2. tools/list -- and what the schemas cost (ch:mcp-schemas)")
tl = call("tools/list")["result"]["tools"]
blob = json.dumps(tl, separators=(",", ":"))
for t in tl:
    print("   %-18s readOnly=%-5s idempotent=%s"
          % (t["name"], t["annotations"]["readOnlyHint"],
             t["annotations"]["idempotentHint"]))
print("   listing size: %d bytes, roughly %d tokens, paid every request"
      % (len(blob), len(blob) // 4))
print()

print("3. tools/call -- a normal call")
r = call("tools/call", {"name": "convert_currency",
                        "arguments": {"amount": 120, "frm": "USD",
                                      "to": "EUR"}})
print("   %s" % r["result"]["content"][0]["text"])
print("   structured: %s" % json.dumps(r["result"]["structuredContent"]))
print()

print("4. a bad enum value -- the error names the field and lists valid values")
r = call("tools/call", {"name": "convert_currency",
                        "arguments": {"amount": 5, "frm": "USD", "to": "JPY"}})
print("   %s" % r["error"]["message"])
print()

print("5. a missing argument -- named, not merely rejected")
r = call("tools/call", {"name": "record_expense",
                        "arguments": {"amount": 12}})
print("   %s" % r["error"]["message"])
print()

print("6. a wrong tool name -- the inventory is enumerated for the retry")
r = call("tools/call", {"name": "convert_money", "arguments": {}})
print("   %s" % r["error"]["message"])
print()

print("7. idempotency -- the same key twice is one effect")
args = {"amount": 42.5, "currency": "EUR", "memo": "taxi",
        "idempotency_key": "run-7:step-3"}
a = call("tools/call", {"name": "record_expense", "arguments": args})
b = call("tools/call", {"name": "record_expense", "arguments": args})
print("   first  -> %s (duplicate=%s)"
      % (a["result"]["content"][0]["text"],
         a["result"]["structuredContent"]["duplicate"]))
print("   replay -> %s (duplicate=%s)"
      % (b["result"]["content"][0]["text"],
         b["result"]["structuredContent"]["duplicate"]))
print("   ledger has %d entry after 2 calls" % len(LEDGER))
print()

print("8. version negotiation -- an unsupported revision lists what IS supported")
r = call("tools/list", version="2027-01-01")
print("   code:      %d" % r["error"]["code"])
print("   message:   %s" % r["error"]["message"])
print("   supported: %s" % ", ".join(r["error"]["data"]["supported"]))
print()

print("9. an older revision inside the window still works")
r = call("tools/list", version="2025-06-18")
print("   tools/list at 2025-06-18 returned %d tools"
      % len(r["result"]["tools"]))
print()

print("10. statelessness -- every request stands alone, so any replica serves")
print("    any request and there is no handshake to lose (ch:mcp-architecture)")
print("    requests issued so far: independent, out of order, no session id")
print()

print("11. the approval snapshot -- hash the definitions (ch:mcp-security)")
print("    tool digest at approval: %s" % tool_digest())

assert len(LEDGER) == 1, "idempotency key failed to suppress the replay"
assert call("tools/list", version="2027-01-01")["error"]["data"]["supported"]
assert d["protocolVersions"] == SUPPORTED
