"""
RoboScene+ — Confidence-Aware 3D Semantic Reconstruction
Gradio demo
"""

import http.server
import json
import os
import socket
import socketserver
import sys
import threading
from pathlib import Path
from urllib.parse import quote

import gradio as gr

# ---------------------------------------------------------------------------
# Resolve paths relative to the repository root (one level up from app/)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
SCENE_GRAPH_PATH = OUTPUTS / "scene_graph.json"

# ---------------------------------------------------------------------------
# Background static file server — serves viewer.html + PLY files
# Gradio 6.x blocks .html from its /file= route, so we need a separate server.
# ---------------------------------------------------------------------------

def _free_port(start: int = 8082) -> int:
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        # Enable SharedArrayBuffer for the sort worker in GaussianSplats3D.js
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()

    def log_message(self, *args):  # silence access logs
        pass


def _launch_file_server() -> int:
    port = _free_port()

    class _Server(socketserver.TCPServer):
        allow_reuse_address = True

    httpd = _Server(("127.0.0.1", port), _QuietHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return port


_FILE_PORT: int = _launch_file_server()
_FILE_BASE: str = f"http://127.0.0.1:{_FILE_PORT}"

# ---------------------------------------------------------------------------
# Optionally import query_scene from the repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(ROOT))
try:
    from query_scene import query_scene, EXAMPLE_QUERIES
    QUERY_AVAILABLE = True
except ImportError:
    QUERY_AVAILABLE = False
    EXAMPLE_QUERIES = [
        "Where is the chair?",
        "What objects are near the window?",
        "Which areas have low confidence?",
        "Describe the room layout.",
    ]

# ---------------------------------------------------------------------------
# Asset paths
# ---------------------------------------------------------------------------
IMG_NAV_MAP      = str(OUTPUTS / "navigability_map.png")
IMG_OBJ_2D       = str(OUTPUTS / "object_positions_2d.png")
IMG_DEAD_ZONE    = str(OUTPUTS / "dead_zones" / "dead_zone_summary.png")
IMG_SEMANTIC     = str(OUTPUTS / "semantic" / "frame_0001_debug.png")

# ── 3D viewer paths ──────────────────────────────────────────────────────────
VIEWER_HTML    = Path(__file__).resolve().parent / "static" / "viewer.html"
# Prefer pruned splat (41MB, ~10s load) over full splat (133MB, ~90s load)
PLY_SEMANTIC   = OUTPUTS / "scene_semantic_pruned.splat"
if not PLY_SEMANTIC.exists():
    PLY_SEMANTIC = OUTPUTS / "scene_semantic.ply"
PLY_APPEARANCE = OUTPUTS / "splat_mast3r_v2" / "scene.ply"


def _viewer_iframe(ply_abs: Path, height: int = 600) -> str:
    """Build an <iframe> + switch-buttons block for the 3D viewer.

    Uses the background static server (_FILE_BASE) instead of Gradio's
    /file= route, which blocks .html files in Gradio 6.x.
    """
    viewer_exists = VIEWER_HTML.exists()
    # Accept any of: pruned .splat, full .splat, or .ply
    sem_candidates = [
        OUTPUTS / "scene_semantic_pruned.splat",
        OUTPUTS / "scene_semantic.splat",
        OUTPUTS / "scene_semantic.ply",
    ]
    sem_file = next((p for p in sem_candidates if p.exists()), None)

    if not viewer_exists or sem_file is None:
        missing = "scene_semantic_pruned.splat / scene_semantic.splat / scene_semantic.ply"
        return (
            "<div style='padding:20px;background:#1a1a2e;border-radius:8px;"
            "color:#888;font-family:monospace;'>"
            "⚠️ <b>3D viewer files not found.</b><br>"
            f"Expected viewer: <code>{VIEWER_HTML}</code><br>"
            f"Expected scene: <code>outputs/{missing}</code><br>"
            "Run <code>scripts/paint_semantic_gaussians.py</code> then "
            "<code>scripts/convert_to_splat.py</code> first."
            "</div>"
        )

    # All URLs go through the background file server (port _FILE_PORT)
    viewer_url = f"{_FILE_BASE}/app/static/viewer.html"
    # Pass the .ply URL — the viewer JS will auto-upgrade to _pruned.splat or .splat
    sem_url = f"{_FILE_BASE}/outputs/scene_semantic.ply"
    app_url = (
        f"{_FILE_BASE}/outputs/splat_mast3r_v2/scene.ply"
        if PLY_APPEARANCE.exists() else sem_url
    )

    # Hash fragment carries the PLY URL (not a query param — avoids router confusion)
    sem_src = f"{viewer_url}#ply={quote(sem_url, safe='')}"
    app_src = f"{viewer_url}#ply={quote(app_url, safe='')}"

    return f"""
<div style="border-radius:10px;overflow:hidden;background:#0f0f1a;
            border:1px solid #2e2e4e;">
  <iframe id="splat-frame"
    src="{sem_src}"
    width="100%"
    height="{height}px"
    frameborder="0"
    allow="accelerometer;gyroscope;xr-spatial-tracking;fullscreen"
    style="display:block;">
  </iframe>
  <div style="padding:10px 14px;display:flex;gap:10px;align-items:center;
              background:#0f0f1a;border-top:1px solid #2e2e4e;">
    <button
      onclick="document.getElementById('splat-frame').src='{sem_src}'"
      style="background:#7F77DD;color:#fff;border:none;padding:8px 18px;
             border-radius:6px;cursor:pointer;font-weight:600;font-size:13px;">
      Semantic View
    </button>
    <button
      onclick="document.getElementById('splat-frame').src='{app_src}'"
      style="background:transparent;color:#7F77DD;border:1px solid #7F77DD;
             padding:8px 18px;border-radius:6px;cursor:pointer;font-size:13px;">
      Appearance View
    </button>
    <span style="color:#555;font-size:11px;margin-left:4px;">
      WASD · drag to look · scroll to zoom · <b>W</b> to fly inside the scene
    </span>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Helper — load scene graph once at startup
# ---------------------------------------------------------------------------
def load_scene_graph() -> dict:
    if not SCENE_GRAPH_PATH.exists():
        return {}
    with open(SCENE_GRAPH_PATH, "r") as fh:
        return json.load(fh)


SCENE_GRAPH: dict = load_scene_graph()


# ---------------------------------------------------------------------------
# Tab 1 helpers — room summary markdown
# ---------------------------------------------------------------------------
def _get_objects(sg: dict) -> list:
    """Return the object list regardless of whether the key is 'objects' or 'nodes'."""
    return sg.get("objects", sg.get("nodes", []))


def build_room_summary(sg: dict) -> str:
    if not sg:
        return (
            "> ⚠️ **`scene_graph.json` not found.** "
            "Run the full pipeline first to generate scene data."
        )

    objs  = _get_objects(sg)
    n_obj = len(objs)

    # ── Room dimensions ─────────────────────────────────────────────────
    # Support both legacy {metadata: {dimensions: ...}} and the current
    # {room_summary: {room_dimensions_m: [W, H, D]}} schemas.
    rs    = sg.get("room_summary", {})
    meta  = sg.get("metadata", sg.get("room_metadata", {}))

    dims_list = rs.get("room_dimensions_m", [])
    if dims_list and len(dims_list) >= 3:
        width, height, depth = [round(v, 2) for v in dims_list[:3]]
    else:
        dims_dict = meta.get("room_dimensions", meta.get("dimensions", {}))
        width  = dims_dict.get("width_m",  dims_dict.get("width",  "?"))
        depth  = dims_dict.get("depth_m",  dims_dict.get("depth",  "?"))
        height = dims_dict.get("height_m", dims_dict.get("height", "?"))

    # ── Coverage ─────────────────────────────────────────────────────────
    coverage = (
        rs.get("navigability_coverage_pct")
        or meta.get("coverage_percentage")
        or meta.get("scene_coverage_pct")
        or meta.get("coverage_pct", "?")
    )

    # ── Provenance counts (current schema) ───────────────────────────────
    n_sparse   = rs.get("num_sparse_objects", "?")
    n_inferred = rs.get("num_inferred_objects", "?")

    # ── Average confidence ───────────────────────────────────────────────
    confidences = [
        o.get("reconstruction_confidence",
              o.get("confidence", o.get("confidence_score", None)))
        for o in objs
        if o.get("reconstruction_confidence",
                 o.get("confidence", o.get("confidence_score", None))) is not None
    ]
    avg_conf = (
        f"{sum(confidences) / len(confidences):.2f}"
        if confidences else "N/A"
    )

    lines = [
        "### 🏠 Room Summary",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| **Dimensions** | {width} m × {height} m × {depth} m |",
        f"| **Objects detected** | {n_obj} |",
        f"| **Scene coverage** | {coverage}% |",
        f"| **Avg. object confidence** | {avg_conf} |",
        f"| **Sparse objects** | {n_sparse} |",
        f"| **Inferred objects** | {n_inferred} |",
        "",
    ]

    if n_obj:
        lines += [
            "**Object inventory:**",
            "",
        ]
        for o in objs[:20]:   # cap preview at 20
            name = o.get("label", o.get("class", o.get("name", "unknown")))
            conf = o.get("reconstruction_confidence",
                         o.get("confidence", o.get("confidence_score", 0.0)))
            prov = o.get("provenance", "")
            prov_tag = f" _{prov}_" if prov else ""
            emoji = "🟢" if conf > 0.7 else ("🟡" if conf >= 0.3 else "🔴")
            lines.append(f"- {emoji} **{name}** (conf: {conf:.2f}){prov_tag}")
        if n_obj > 20:
            lines.append(f"- … and {n_obj - 20} more")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 2 helpers — object dataframe
# ---------------------------------------------------------------------------
def _conf_badge(conf: float) -> str:
    if conf > 0.7:
        return f"🟢 {conf:.2f}"
    elif conf >= 0.3:
        return f"🟡 {conf:.2f}"
    return f"🔴 {conf:.2f}"


def build_object_table(sg: dict):
    """Return (headers, rows) for gr.Dataframe."""
    headers = ["Object", "Position (3D)", "Confidence", "Provenance", "Volume (m³)"]
    if not sg:
        return headers, [["—", "—", "—", "—", "—"]]

    rows = []
    for o in _get_objects(sg):
        name = o.get("label", o.get("class", o.get("name", "unknown")))

        # current schema stores reconstruction_confidence; fall back to others
        conf = float(
            o.get("reconstruction_confidence",
                  o.get("confidence", o.get("confidence_score", 0.0)))
        )

        # position_3d may be a list [x,y,z] or a dict {x,y,z}
        pos_raw = o.get("position_3d",
                  o.get("position",
                  o.get("centroid_3d", [])))
        if isinstance(pos_raw, (list, tuple)) and len(pos_raw) >= 3:
            pos_str = f"({pos_raw[0]:.2f}, {pos_raw[1]:.2f}, {pos_raw[2]:.2f})"
        elif isinstance(pos_raw, dict):
            x = pos_raw.get("x", pos_raw.get("X", "?"))
            y = pos_raw.get("y", pos_raw.get("Y", "?"))
            z = pos_raw.get("z", pos_raw.get("Z", "?"))
            pos_str = (
                f"({x:.2f}, {y:.2f}, {z:.2f})"
                if all(isinstance(v, (int, float)) for v in [x, y, z])
                else f"({x}, {y}, {z})"
            )
        else:
            pos_str = str(pos_raw) if pos_raw else "N/A"

        prov    = o.get("provenance", o.get("source", "MASt3R + SAM2"))
        vol     = o.get("volume_m3", o.get("volume", "N/A"))
        vol_str = f"{vol:.3f}" if isinstance(vol, (int, float)) else str(vol)

        rows.append([name, pos_str, _conf_badge(conf), prov, vol_str])

    return headers, rows if rows else [["—", "—", "—", "—", "—"]]


# ---------------------------------------------------------------------------
# Tab 3 helpers — robot query
# ---------------------------------------------------------------------------
def run_robot_query(question: str) -> str:
    if not question or not question.strip():
        return "⚠️ Please enter a question first."

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return (
            "❌ **ANTHROPIC_API_KEY not set.**\n\n"
            "Export it in your shell:\n"
            "```\nexport ANTHROPIC_API_KEY='sk-ant-...'\n```\n"
            "Then restart the Gradio server."
        )

    if not QUERY_AVAILABLE:
        return (
            "❌ `query_scene.py` could not be imported. "
            "Make sure it is present in the repository root and all "
            "its dependencies are installed."
        )

    if not SCENE_GRAPH:
        return (
            "❌ `scene_graph.json` not found. "
            "Run the full pipeline to generate the scene graph first."
        )

    try:
        answer = query_scene(question, SCENE_GRAPH)
        return answer
    except Exception as exc:
        return f"❌ Query failed: {exc}"


# ---------------------------------------------------------------------------
# Safe image loader
# ---------------------------------------------------------------------------
def safe_img(path: str):
    """Return path if file exists, else None (Gradio shows placeholder)."""
    return path if Path(path).exists() else None


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────────────────── */
:root {
    --bg-main:    #0f0f1a;
    --bg-card:    #1a1a2e;
    --accent:     #7F77DD;
    --accent-dim: #5a54aa;
    --text:       #e0e0e0;
    --text-muted: #888aaa;
    --border:     #2e2e4e;
    --radius:     10px;
}

body, .gradio-container {
    background: var(--bg-main) !important;
    color:      var(--text)    !important;
    font-family: system-ui, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif !important;
}

/* ── Header / hero ──────────────────────────────────────────── */
#roboscene-header {
    text-align: center;
    padding: 28px 20px 12px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
}
#roboscene-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), #a09cf7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
#roboscene-header h3 {
    color: var(--text-muted);
    font-weight: 400;
    font-size: 1rem;
}

/* ── Tabs ───────────────────────────────────────────────────── */
.tabs > .tab-nav {
    background:    var(--bg-card)  !important;
    border-bottom: 1px solid var(--border) !important;
}
.tabs > .tab-nav button {
    color:       var(--text-muted) !important;
    font-size:   0.95rem !important;
    padding:     10px 18px !important;
    border-radius: var(--radius) var(--radius) 0 0 !important;
    transition:  all 0.2s ease;
}
.tabs > .tab-nav button.selected,
.tabs > .tab-nav button:hover {
    color:            var(--accent) !important;
    background:       var(--bg-main) !important;
    border-bottom:    2px solid var(--accent) !important;
}

/* ── Cards / panels ─────────────────────────────────────────── */
.card {
    background:    var(--bg-card) !important;
    border:        1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding:       20px !important;
}
.gr-panel, .gr-box, fieldset {
    background:    var(--bg-card) !important;
    border:        1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Images ─────────────────────────────────────────────────── */
.gr-image, img {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
}

/* ── Inputs / textboxes ─────────────────────────────────────── */
input[type="text"], textarea, .gr-textbox textarea {
    background:    var(--bg-card)  !important;
    color:         var(--text)     !important;
    border:        1px solid var(--border) !important;
    border-radius: 6px !important;
}
input[type="text"]:focus, textarea:focus {
    border-color:  var(--accent) !important;
    outline:       none !important;
    box-shadow:    0 0 0 2px rgba(127,119,221,0.25) !important;
}

/* ── Buttons ─────────────────────────────────────────────────── */
.gr-button-primary, button.primary {
    background:    var(--accent)   !important;
    color:         #fff            !important;
    border:        none            !important;
    border-radius: 6px !important;
    font-weight:   600 !important;
    transition:    background 0.2s ease;
}
.gr-button-primary:hover, button.primary:hover {
    background: var(--accent-dim) !important;
}
button.secondary {
    background:    transparent !important;
    color:         var(--accent)  !important;
    border:        1px solid var(--accent) !important;
    border-radius: 6px !important;
    transition:    all 0.2s ease;
}
button.secondary:hover {
    background: rgba(127,119,221,0.12) !important;
}

/* ── Dataframe ──────────────────────────────────────────────── */
.gr-dataframe table {
    background: var(--bg-card) !important;
    color:      var(--text)    !important;
}
.gr-dataframe thead th {
    background:  var(--bg-main) !important;
    color:       var(--accent)  !important;
    font-weight: 600 !important;
}
.gr-dataframe tbody tr:hover td {
    background: rgba(127,119,221,0.08) !important;
}

/* ── Markdown ──────────────────────────────────────────────── */
.gr-markdown h3 { color: var(--accent); }
.gr-markdown table {
    border-collapse: collapse;
    width: 100%;
}
.gr-markdown th, .gr-markdown td {
    border:  1px solid var(--border);
    padding: 8px 12px;
}
.gr-markdown th { background: var(--bg-main); color: var(--accent); }

/* ── Labels ─────────────────────────────────────────────────── */
label, .gr-label {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
}

/* ── Scrollbars ─────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-dim); }
"""

# ---------------------------------------------------------------------------
# Pipeline stage descriptions
# ---------------------------------------------------------------------------
PIPELINE_STAGES = [
    {
        "step": "1",
        "name": "📹 Video Ingestion",
        "tool": "OpenCV / ffmpeg",
        "produces": "Extracted frames (`.jpg`) at configurable FPS, "
                    "ready for downstream reconstruction.",
    },
    {
        "step": "2",
        "name": "🗺️ MASt3R-SLAM",
        "tool": "MASt3R (Meta AI)",
        "produces": "Dense point cloud + per-frame camera poses "
                    "(`.ply` + `poses.json`). Metric-scale 3D structure "
                    "from monocular video.",
    },
    {
        "step": "3",
        "name": "✨ Gaussian Splatting",
        "tool": "3D-GS / Nerfstudio",
        "produces": "Photorealistic radiance-field representation "
                    "(`.splat`). Enables novel-view synthesis and "
                    "high-fidelity scene visualisation.",
    },
    {
        "step": "4",
        "name": "🏷️ Grounded SAM 2",
        "tool": "GroundingDINO + SAM 2 (Meta)",
        "produces": "Per-frame instance masks + class labels "
                    "(`frame_XXXX_debug.png`). Open-vocabulary "
                    "detection without a fixed class list.",
    },
    {
        "step": "5",
        "name": "📐 3D Lifting",
        "tool": "Depth unprojection + pose fusion",
        "produces": "Per-object 3D bounding boxes, centroids, and "
                    "volumes in world coordinates. Multi-frame detections "
                    "are fused and outliers pruned.",
    },
    {
        "step": "6",
        "name": "🎯 Confidence Map",
        "tool": "Ray-casting + coverage analysis",
        "produces": "Bird's-eye navigability map (`navigability_map.png`) "
                    "— green = well observed, amber = sparse, "
                    "red = dead zone. Dead-zone completion via inpainting.",
    },
    {
        "step": "7",
        "name": "🧠 Scene Graph + Claude API",
        "tool": "NetworkX + Anthropic Claude",
        "produces": "Structured `scene_graph.json` with spatial relations, "
                    "confidence scores, and provenance. Natural-language "
                    "queries answered by Claude with scene-graph context.",
    },
]


def build_pipeline_markdown() -> str:
    lines = [
        "## 🔧 Seven-Stage Reconstruction Pipeline",
        "",
        "RoboScene+ converts a raw monocular video into a queryable, "
        "confidence-annotated 3D semantic scene graph.",
        "",
    ]
    for stage in PIPELINE_STAGES:
        lines += [
            f"---",
            f"### Stage {stage['step']} — {stage['name']}",
            f"**Tool / Model:** `{stage['tool']}`  ",
            f"**Produces:** {stage['produces']}",
            "",
        ]
    lines += [
        "---",
        "",
        "> **Confidence scoring** propagates through every stage: "
        "observation density (MASt3R), mask quality (SAM2 IoU), "
        "and multi-frame fusion agreement all contribute to the "
        "per-object confidence stored in `scene_graph.json`.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.purple,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#0f0f1a",
    body_text_color="#e0e0e0",
    block_background_fill="#1a1a2e",
    block_border_color="#2e2e4e",
    block_label_text_color="#888aaa",
    input_background_fill="#1a1a2e",
    button_primary_background_fill="#7F77DD",
    button_primary_background_fill_hover="#5a54aa",
    button_secondary_background_fill="transparent",
    button_secondary_border_color="#7F77DD",
    button_secondary_text_color="#7F77DD",
)


def build_app() -> gr.Blocks:
    # Pre-compute static data
    room_summary_md = build_room_summary(SCENE_GRAPH)
    obj_headers, obj_rows = build_object_table(SCENE_GRAPH)
    pipeline_md = build_pipeline_markdown()

    with gr.Blocks(
        title="RoboScene+",
        analytics_enabled=False,
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────
        with gr.Row(elem_id="roboscene-header"):
            gr.Markdown(
                """
# RoboScene+ 🤖
### Confidence-Aware 3D Semantic Reconstruction | UCL MEng Robotics & AI
"""
            )

        # ── Tabs ─────────────────────────────────────────────────────────
        with gr.Tabs():

            # ════════════════════════════════════════════════════════════
            # TAB 1 — 3D Scene Viewer (PRIMARY)
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("🔍 3D Scene Viewer"):
                gr.HTML(_viewer_iframe(PLY_SEMANTIC, height=620),
                        sanitize_html=False)

                with gr.Row():
                    # Semantic class legend
                    gr.Markdown(
                        """
**Semantic classes:**
🟥 bed &nbsp;&nbsp; 🟦 desk &nbsp;&nbsp; 🔵 chair &nbsp;&nbsp; 💙 laptop
&nbsp;&nbsp; 🟩 monitor &nbsp;&nbsp; 🟠 fan &nbsp;&nbsp; 🟨 lamp
&nbsp;&nbsp; 🟫 shelf &nbsp;&nbsp; 🌫️ door &nbsp;&nbsp; 🩵 window

_Use **Semantic View** to see labeled objects · **Appearance View** for the
photorealistic reconstruction · Press **W** inside the viewer to fly into the scene._
"""
                    )

                gr.Markdown("---")
                gr.Markdown(room_summary_md)

            # ════════════════════════════════════════════════════════════
            # TAB 2 — Object Map
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("📊 Object Map"):
                gr.Markdown("### 📦 Detected Objects — Full Inventory")
                gr.Markdown(
                    "Confidence legend: 🟢 > 0.70 (high) · "
                    "🟡 0.30–0.70 (medium) · 🔴 < 0.30 (low)"
                )

                gr.Dataframe(
                    value=obj_rows,
                    headers=obj_headers,
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=False,
                    row_count=(min(len(obj_rows) + 1, 20), "dynamic"),
                )

                gr.Markdown("---")
                gr.Markdown("### 🕳️ Dead Zone Analysis")

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Image(
                            value=safe_img(IMG_DEAD_ZONE),
                            label="Dead Zone Summary",
                            show_label=True,
                            type="filepath",
                            interactive=False,
                        )
                    with gr.Column(scale=1):
                        gr.Markdown(
                            """
**What are dead zones?**

Dead zones are regions of the scene that were never
directly observed by the camera — occluded areas, narrow
gaps, or regions outside the video's field of view.

RoboScene+ uses **ray-cast coverage analysis** to detect
these regions and marks them with low confidence (🔴).

**Completion strategy:**  
Unobserved cells are inpainted using a weighted average
of neighbouring observed cells, and flagged with
`provenance = "inpainted"` in the scene graph so
downstream planners can treat them with extra caution.
"""
                        )

            # ════════════════════════════════════════════════════════════
            # TAB 3 — Robot Query
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("🤖 Robot Query"):
                gr.Markdown(
                    """
### 🧠 Ask a Question About the 3D Scene

The robot's scene graph is loaded into context and passed to
**Claude** (Anthropic) so you can query the 3D environment in
plain English.  Results include object locations, confidence
levels, spatial relationships, and coverage statistics.
"""
                )

                # Example query buttons
                gr.Markdown("**Quick examples — click to fill the input:**")

                query_input = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. Where is the chair?",
                    lines=2,
                    interactive=True,
                )

                with gr.Row():
                    example_btns = []
                    for eq in EXAMPLE_QUERIES[:4]:
                        btn = gr.Button(eq, variant="secondary", size="sm")
                        example_btns.append(btn)

                # Wire each example button to fill the textbox
                for btn in example_btns:
                    btn.click(
                        fn=lambda q=btn.value: q,
                        inputs=[],
                        outputs=[query_input],
                    )

                ask_btn = gr.Button("Ask Robot 🤖", variant="primary", size="lg")

                query_output = gr.Textbox(
                    label="Robot Response",
                    lines=8,
                    interactive=False,
                    placeholder="The robot's answer will appear here…",
                )

                ask_btn.click(
                    fn=run_robot_query,
                    inputs=[query_input],
                    outputs=[query_output],
                )
                query_input.submit(
                    fn=run_robot_query,
                    inputs=[query_input],
                    outputs=[query_output],
                )

                if not os.environ.get("ANTHROPIC_API_KEY"):
                    gr.Markdown(
                        "> ⚠️ **`ANTHROPIC_API_KEY` is not set.** "
                        "Robot queries will return an error until you "
                        "export the key and restart the server."
                    )

            # ════════════════════════════════════════════════════════════
            # TAB 4 — Pipeline
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("⚙️ Pipeline"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown(pipeline_md)

                    with gr.Column(scale=2):
                        gr.Markdown("### 🔬 Example Semantic Segmentation Output")
                        gr.Image(
                            value=safe_img(IMG_SEMANTIC),
                            label="Grounded SAM 2 — frame_0001_debug.png",
                            show_label=True,
                            type="filepath",
                            interactive=False,
                        )
                        gr.Markdown(
                            "_Per-frame debug visualisation produced by "
                            "**Grounded SAM 2** (Stage 4). Coloured instance "
                            "masks are overlaid on the source frame with "
                            "class labels and confidence scores._"
                        )

        # ── Footer ───────────────────────────────────────────────────────
        gr.Markdown(
            """
---
<div style="text-align:center; color:#888aaa; font-size:0.8rem; padding: 8px 0 4px;">
RoboScene+ · UCL MEng Robotics & AI · Built with
<a href="https://github.com/gradio-app/gradio" style="color:#7F77DD;">Gradio</a>,
<a href="https://www.anthropic.com" style="color:#7F77DD;">Claude</a> &
<a href="https://github.com/facebookresearch/segment-anything-2" style="color:#7F77DD;">SAM 2</a>
</div>
""",
            sanitize_html=False,
        )

    return demo


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_PORT", 7860)),
        share=False,
        show_error=True,
        theme=THEME,
        css=CUSTOM_CSS,
        allowed_paths=[
            str(Path(__file__).resolve().parent),   # app/ (serves viewer.html)
            str(ROOT / "outputs"),                  # outputs/ (serves PLY files)
        ],
    )