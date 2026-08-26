# -*- coding: utf-8 -*-
# Extracted from: Chapter 95 — Function Calling and Tool Use
# Source: src/.../ch095-function-calling.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A complete tool-calling loop, with the four failure modes distinguished."""
import json

TOOLS = {
    "get_weather": {
        "description": "Current weather for a city.",
        "params": {"city": str, "units": str},
        "required": ["city"],
        "enum": {"units": ["celsius", "fahrenheit"]},
    },
    "search_docs": {
        "description": "Full-text search over the internal documentation.",
        "params": {"query": str, "limit": int},
        "required": ["query"],
        "enum": {},
    },
    "calculate": {
        "description": "Evaluate an arithmetic expression.",
        "params": {"expression": str},
        "required": ["expression"],
        "enum": {},
    },
}


def validate_call(call):
    """Returns (ok, message). This is the parse+validate box of fig:tool-loop."""
    if not isinstance(call, dict) or "name" not in call:
        return False, "malformed: no 'name' field"
    name = call["name"]
    if name not in TOOLS:
        return False, (f"hallucinated tool {name!r}; "
                       f"available: {sorted(TOOLS)}")
    spec = TOOLS[name]
    args = call.get("arguments", {})
    if not isinstance(args, dict):
        return False, "malformed: 'arguments' is not an object"
    for req in spec["required"]:
        if req not in args:
            return False, f"missing required parameter {req!r}"
    for key, val in args.items():
        if key not in spec["params"]:
            return False, f"unknown parameter {key!r} for {name}"
        want = spec["params"][key]
        if not isinstance(val, want):
            return False, (f"parameter {key!r} must be {want.__name__}, "
                           f"received {type(val).__name__}")
        if key in spec["enum"] and val not in spec["enum"][key]:
            return False, (f"parameter {key!r} must be one of "
                           f"{spec['enum'][key]}, received {val!r}")
    return True, "ok"


CANDIDATE_CALLS = [
    ({"name": "get_weather", "arguments": {"city": "Paris"}}, "correct"),
    ({"name": "get_wether", "arguments": {"city": "Paris"}}, "hallucinated tool"),
    ({"name": "get_weather", "arguments": {}}, "missing required"),
    ({"name": "get_weather", "arguments": {"city": 42}}, "wrong type"),
    ({"name": "get_weather", "arguments": {"city": "Paris",
                                           "units": "kelvin"}}, "bad enum"),
    ({"name": "search_docs", "arguments": {"query": "weather in Paris"}},
     "WRONG TOOL — valid, and not what was asked"),
]

print(f"{'call':<52} {'valid':>7}  diagnosis")
for call, label in CANDIDATE_CALLS:
    ok, msg = validate_call(call)
    shown = json.dumps(call)[:50]
    print(f"{shown:<52} {str(ok):>7}  {msg if not ok else label}")

print("""
The last row is the important one. It passes every structural check — real tool,
required parameter present, correct types — and it is the wrong tool for the
question. No validator catches that, because the call is well-formed and wrong
only relative to intent.

That is tbl:tool-failure-modes' division: the first four rows are structural and
a grammar makes them unreachable; the last is semantic and survives.""")


# --- the loop itself -------------------------------------------------------
def execute(call):
    """Your code. Returns a STRING result, including for errors — the model
    can only see what is put back into its context."""
    name, args = call["name"], call.get("arguments", {})
    if name == "get_weather":
        if args["city"] not in ("Paris", "London"):
            return f"ERROR: unknown city {args['city']!r}; try Paris or London"
        return f"{args['city']}: 14C, overcast"
    if name == "calculate":
        expr = args["expression"]
        if not set(expr) <= set("0123456789+-*/(). "):
            return f"ERROR: expression contains disallowed characters"
        try:
            return f"result: {eval(expr, {'__builtins__': {}}, {})}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
    if name == "search_docs":
        return f"3 results for {args['query']!r}"
    return "ERROR: not implemented"


def dispatch_loop(scripted_responses, max_iterations=6, verbose=True):
    """Equation (eq:dispatch-loop). `scripted_responses` stands in for the
    model so the loop's behaviour is deterministic and inspectable."""
    context, rounds = [], 0
    for step, response in enumerate(scripted_responses):
        if response.get("final"):
            if verbose:
                print(f"  [{step}] final answer: {response['final']}")
            return response["final"], rounds, context
        rounds += 1
        if rounds > max_iterations:
            return "ERROR: iteration limit", rounds, context
        call = response["call"]
        ok, msg = validate_call(call)
        result = execute(call) if ok else f"ERROR: {msg}"
        context.append({"call": call, "result": result})
        if verbose:
            print(f"  [{step}] call {call['name']}({call.get('arguments')}) "
                  f"-> {result}")
    return "ERROR: ran out of scripted responses", rounds, context


print("\nA loop where the model gets it wrong, sees the error, and recovers:\n")
answer, rounds, ctx = dispatch_loop([
    {"call": {"name": "get_weather", "arguments": {"city": "Pariss"}}},
    {"call": {"name": "get_weather", "arguments": {"city": "Paris"}}},
    {"final": "It is 14C and overcast in Paris."},
])
print(f"\n  rounds used: {rounds}, context entries: {len(ctx)}")
print("The error string went back into the context, and the next call was "
      "corrected. That recovery is only possible because execute() RETURNED "
      "the error rather than raising it.")
