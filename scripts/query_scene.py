"""Natural language scene query interface over scene_graph.json using the Anthropic Claude API."""

import argparse
import json
import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Example queries (4 presets — used by Gradio buttons in Session 9)
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLE_QUERIES: list[str] = [
    "Where is the laptop?",
    "What objects are low confidence?",
    "Describe the room layout for a robot entering through the door",
    "What is the safest path from the door to the desk?",
]

# ─────────────────────────────────────────────────────────────────────────────
# Default paths / model
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SCENE_GRAPH = "outputs/scene_graph.json"
DEFAULT_MODEL       = "claude-sonnet-4-5"
MAX_TOKENS          = 500

# ─────────────────────────────────────────────────────────────────────────────
# System prompt template
# Embedded verbatim as specified — {scene_graph_json} is substituted at runtime
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are a robot spatial reasoning assistant for a humanoid robot.
You have access to a 3D reconstruction of an indoor room.
Here is the complete scene graph: {scene_graph_json}
When answering navigation questions always:
- Give 3D coordinates as (X.Xm, Y.Ym, Z.Zm) from room origin
- Cite reconstruction_confidence and explain what it means:
>0.7=well observed, 0.3-0.7=partially seen, <0.3=uncertain
- For path questions list waypoints and their confidence
- Flag any low-confidence regions on the path
Be concise and precise — you are feeding a robot planner."""


# ─────────────────────────────────────────────────────────────────────────────
# Scene graph loading
# ─────────────────────────────────────────────────────────────────────────────

def load_scene_graph(scene_graph_path: str) -> dict:
    """
    Load scene_graph.json.  If it does not exist, attempt to build it
    on-the-fly by calling build_scene_graph.py's public function.
    """
    p = Path(scene_graph_path)
    if p.exists():
        with open(p) as f:
            return json.load(f)

    # Try to build it automatically
    print(f"  Scene graph not found at {scene_graph_path}. "
          "Attempting to build it …")
    try:
        # Add scripts/ to path so we can import sibling module
        scripts_dir = Path(__file__).parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from build_scene_graph import build_scene_graph  # type: ignore
        graph = build_scene_graph(output_path=scene_graph_path)
        return graph
    except Exception as exc:
        sys.exit(
            f"ERROR: scene graph not found at {scene_graph_path} and "
            f"could not build it automatically ({exc}).\n"
            "Run: python scripts/build_scene_graph.py"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic client initialisation (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

_client = None   # module-level singleton


def _get_client():
    """
    Return the Anthropic client, initialising it on first call.
    Reads ANTHROPIC_API_KEY from the environment — never from code.
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "ERROR: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it with:  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "Or add it to your .env file and run:  export $(cat .env | xargs)"
        )

    try:
        import anthropic  # type: ignore
    except ImportError:
        sys.exit(
            "ERROR: anthropic package not installed.\n"
            "Install with:  pip install anthropic"
        )

    _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Core query function (importable by Gradio app)
# ─────────────────────────────────────────────────────────────────────────────

def query_scene(question: str,
                scene_graph: dict | None = None,
                scene_graph_path: str = DEFAULT_SCENE_GRAPH,
                model: str = DEFAULT_MODEL,
                stream: bool = True) -> str:
    """
    Send a natural-language question about the scene to Claude.

    Parameters
    ----------
    question         : The user's question.
    scene_graph      : Pre-loaded scene graph dict (optional — avoids re-loading).
    scene_graph_path : Path to scene_graph.json (used if scene_graph is None).
    model            : Anthropic model ID.
    stream           : If True, print tokens to stdout as they arrive and
                       return the full concatenated response string.
                       If False, return the full response silently.

    Returns
    -------
    str : The assistant's response.
    """
    # Load scene graph if not supplied
    if scene_graph is None:
        scene_graph = load_scene_graph(scene_graph_path)

    # Serialise graph compactly (no indent) to keep token count low
    scene_graph_json = json.dumps(scene_graph, separators=(",", ":"))

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        scene_graph_json=scene_graph_json
    )

    client = _get_client()

    if stream:
        # Streaming — print tokens live and accumulate full text
        full_response = ""
        with client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        ) as stream_ctx:
            for text_chunk in stream_ctx.text_stream:
                print(text_chunk, end="", flush=True)
                full_response += text_chunk
        print()   # newline after streamed output
        return full_response
    else:
        # Non-streaming — used when caller handles display (e.g. Gradio streaming)
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        )
        return response.content[0].text


def query_scene_streaming(question: str,
                          scene_graph: dict | None = None,
                          scene_graph_path: str = DEFAULT_SCENE_GRAPH,
                          model: str = DEFAULT_MODEL):
    """
    Generator variant for Gradio streaming:  yields partial response strings.

    Usage in Gradio:
        gr.ChatInterface(fn=query_scene_streaming, ...)

    Example:
        for chunk in query_scene_streaming("Where is the desk?"):
            print(chunk, end="")
    """
    if scene_graph is None:
        scene_graph = load_scene_graph(scene_graph_path)

    scene_graph_json = json.dumps(scene_graph, separators=(",", ":"))
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        scene_graph_json=scene_graph_json
    )

    client = _get_client()
    accumulated = ""

    with client.messages.stream(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    ) as stream_ctx:
        for text_chunk in stream_ctx.text_stream:
            accumulated += text_chunk
            yield accumulated   # Gradio expects cumulative string


# ─────────────────────────────────────────────────────────────────────────────
# Interactive CLI loop
# ─────────────────────────────────────────────────────────────────────────────

def _print_welcome(scene_graph: dict) -> None:
    """Print a startup banner with scene summary."""
    summary = scene_graph.get("room_summary", {})
    nodes   = scene_graph.get("nodes", [])
    edges   = scene_graph.get("edges", [])
    dims    = summary.get("room_dimensions_m", [0, 0, 0])
    nav_pct = summary.get("navigability_coverage_pct", 0.0)

    print()
    print("╔" + "═" * 62 + "╗")
    print("║        RoboScene+ Natural Language Query Interface           ║")
    print("╚" + "═" * 62 + "╝")
    print(f"  Model      : {DEFAULT_MODEL}")
    print(f"  Scene      : {len(nodes)} objects  |  {len(edges)} spatial edges")
    print(f"  Room dims  : {dims[0]:.1f} m × {dims[1]:.1f} m × {dims[2]:.1f} m")
    print(f"  Navigable  : {nav_pct:.1f}% of scene volume")

    print("\n  Objects in scene:")
    for n in sorted(nodes,
                    key=lambda x: x.get("reconstruction_confidence", 0),
                    reverse=True):
        conf = n.get("reconstruction_confidence", 0.0)
        prov = n.get("provenance", "?")
        x, y, z = n.get("position_3d", [0, 0, 0])
        bar_filled = int(conf * 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)
        print(f"    {n['label']:<12}  [{bar}] {conf:.2f}  {prov:<10}  "
              f"({x:.2f}m, {y:.2f}m, {z:.2f}m)")

    print()
    print("  Example queries (type the number or paste your own):")
    for i, q in enumerate(EXAMPLE_QUERIES, 1):
        print(f"    {i}. {q}")

    print()
    print("  Commands:  'quit' or 'exit' to stop")
    print("             '1'–'4'  to run a preset query")
    print("             Any other text → send as query")
    print("─" * 64)


def interactive_loop(scene_graph: dict, model: str = DEFAULT_MODEL) -> None:
    """Run an interactive query loop reading from stdin."""
    _print_welcome(scene_graph)

    while True:
        try:
            print()
            user_input = input("  Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("  Goodbye.")
            break

        # Numeric shortcut → preset query
        if user_input in ("1", "2", "3", "4"):
            question = EXAMPLE_QUERIES[int(user_input) - 1]
            print(f"  → {question}")
        else:
            question = user_input

        print()
        print("  " + "─" * 60)
        print("  Robot Planner Response:")
        print("  " + "─" * 60)

        try:
            query_scene(
                question=question,
                scene_graph=scene_graph,
                model=model,
                stream=True,
            )
        except Exception as exc:
            print(f"\n  ERROR: {exc}")

        print("  " + "─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Natural language query interface for the RoboScene+ scene graph."
    )
    p.add_argument(
        "--scene_graph",
        default=DEFAULT_SCENE_GRAPH,
        help=f"Path to scene_graph.json (default: {DEFAULT_SCENE_GRAPH})",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Anthropic model ID (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--question",
        default=None,
        help="Single question to answer non-interactively (optional)",
    )
    p.add_argument(
        "--list_examples",
        action="store_true",
        help="Print the 4 example queries and exit",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # --list_examples shortcut
    if args.list_examples:
        print("\nExample queries:")
        for i, q in enumerate(EXAMPLE_QUERIES, 1):
            print(f"  {i}. {q}")
        print()
        return

    # Load the API key early so we fail fast before loading the graph
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "ERROR: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it:  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "Or:         export $(cat .env | xargs)"
        )

    # Load scene graph
    print(f"Loading scene graph from {args.scene_graph} …")
    scene_graph = load_scene_graph(args.scene_graph)
    print(f"  ✓  {len(scene_graph.get('nodes', []))} nodes  "
          f"| {len(scene_graph.get('edges', []))} edges loaded.")

    if args.question:
        # Single non-interactive query
        print(f"\nQuestion: {args.question}")
        print("─" * 64)
        query_scene(
            question=args.question,
            scene_graph=scene_graph,
            model=args.model,
            stream=True,
        )
        print("─" * 64)
    else:
        # Full interactive loop
        interactive_loop(scene_graph, model=args.model)


if __name__ == "__main__":
    main()