#!/usr/bin/env python3
"""
build_scene_graph.py  –  Session 8 Part A: Scene Graph Construction
====================================================================
Loads outputs/objects_3d.json (with confidence + provenance fields from
Session 6) and builds a spatial scene graph saved as outputs/scene_graph.json.

Node format:
    {
      "id": "bed",
      "label": "bed",
      "position_3d": [x, y, z],
      "bbox_min": [x, y, z],
      "bbox_max": [x, y, z],
      "volume_m3": float,
      "reconstruction_confidence": float,
      "provenance": "observed"|"sparse"|"inferred",
      "frames_seen": int
    }

Spatial edges computed automatically:
    on_top_of  – A centroid Y > B bbox_max Y AND A XZ within B bbox XZ ± 0.15 m
    next_to    – centroid distance < 0.8 m (not on_top_of)
    near_wall  – centroid within 0.3 m of any scene bbox face
    between    – A XZ centroid lies between B and C XZ centroids ± 0.3 m

Room summary node:
    {
      "room_dimensions_m": [w, h, d],
      "num_objects": 10,
      "num_sparse_objects": N,
      "num_inferred_objects": N,
      "navigability_coverage_pct": 34.1
    }

Usage:
    python scripts/build_scene_graph.py \
        --objects_file  outputs/objects_3d.json \
        --metadata_file outputs/confidence_metadata.json \
        --output        outputs/scene_graph.json
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds (all in metres)
# ─────────────────────────────────────────────────────────────────────────────
ON_TOP_OF_XZ_MARGIN = 0.15   # A XZ centroid must be within B XZ bbox ± this
NEXT_TO_DIST        = 0.80   # max centroid-to-centroid distance for "next_to"
NEAR_WALL_DIST      = 0.30   # max distance from scene bbox face for "near_wall"
BETWEEN_MARGIN      = 0.30   # ± margin for "between" XZ interpolation test

# Labels that represent architectural wall features.
# Objects with these labels are always flagged near_wall;
# other objects within WALL_PROXIMITY_DIST of them are also flagged.
WALL_LABELS       = {"door", "window", "wall", "shelf"}
WALL_PROXIMITY_DIST = 0.80   # metres — how close to a wall-label to count


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_objects(objects_file: str) -> list[dict]:
    """
    Load objects_3d.json.  Handles both dict-of-dicts (keyed by label)
    and list-of-dicts formats.  Returns a normalised list of dicts,
    each guaranteed to have a 'label' key.
    """
    with open(objects_file) as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        obj_list = []
        for key, val in raw.items():
            entry = dict(val)
            entry.setdefault("label", key)
            obj_list.append(entry)
        return obj_list

    # Already a list
    return list(raw)


def load_confidence_metadata(metadata_file: str) -> dict:
    """Load confidence_metadata.json; return empty dict if missing."""
    p = Path(metadata_file)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Node construction
# ─────────────────────────────────────────────────────────────────────────────

def make_node(obj: dict) -> dict:
    """
    Build a scene-graph node from an objects_3d entry.
    Falls back gracefully when optional fields are absent.
    """
    label = obj.get("label", "unknown")

    centroid = obj.get("centroid_3d", [0.0, 0.0, 0.0])
    bbox_min  = obj.get("bbox_min",   [0.0, 0.0, 0.0])
    bbox_max  = obj.get("bbox_max",   [0.0, 0.0, 0.0])

    return {
        "id":                       label,
        "label":                    label,
        "position_3d":              [round(v, 4) for v in centroid],
        "bbox_min":                 [round(v, 4) for v in bbox_min],
        "bbox_max":                 [round(v, 4) for v in bbox_max],
        "volume_m3":                round(float(obj.get("volume_m3",    0.0)), 6),
        "reconstruction_confidence": round(float(obj.get("reconstruction_confidence", 0.0)), 4),
        "provenance":               obj.get("provenance", "inferred"),
        "frames_seen":              int(obj.get("frames_seen", 0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scene bounding box helpers
# ─────────────────────────────────────────────────────────────────────────────

def scene_bbox(nodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the physical scene bounding box from all object bbox corners.
    Used for room_dimensions_m in the room summary.
    Returns (scene_min [3], scene_max [3]).
    """
    all_min = np.array([n["bbox_min"] for n in nodes], dtype=float)
    all_max = np.array([n["bbox_max"] for n in nodes], dtype=float)
    return all_min.min(axis=0), all_max.max(axis=0)


def centroid_cloud_bbox(nodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the bounding box of object *centroids* only.
    Used for proximity-based edge predicates (near_wall, next_to) where
    the noisy open-set bboxes would give misleading extent.
    Returns (c_min [3], c_max [3]).
    """
    centroids = np.array([n["position_3d"] for n in nodes], dtype=float)
    return centroids.min(axis=0), centroids.max(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial edge predicates
# ─────────────────────────────────────────────────────────────────────────────

def _centroid(node: dict) -> np.ndarray:
    return np.array(node["position_3d"], dtype=float)


def _bbox_min(node: dict) -> np.ndarray:
    return np.array(node["bbox_min"], dtype=float)


def _bbox_max(node: dict) -> np.ndarray:
    return np.array(node["bbox_max"], dtype=float)


def is_on_top_of(a: dict, b: dict) -> bool:
    """
    True when:
      (1) A centroid Y  >  B bbox_max Y          (A is above B's top face)
      (2) A centroid X within [B bbox_min X - margin, B bbox_max X + margin]
      (3) A centroid Z within [B bbox_min Z - margin, B bbox_max Z + margin]
    """
    ca = _centroid(a)
    bmin_b = _bbox_min(b)
    bmax_b = _bbox_max(b)

    above_in_y = ca[1] > bmax_b[1]
    within_x   = (bmin_b[0] - ON_TOP_OF_XZ_MARGIN) <= ca[0] <= (bmax_b[0] + ON_TOP_OF_XZ_MARGIN)
    within_z   = (bmin_b[2] - ON_TOP_OF_XZ_MARGIN) <= ca[2] <= (bmax_b[2] + ON_TOP_OF_XZ_MARGIN)

    return bool(above_in_y and within_x and within_z)


def centroid_dist(a: dict, b: dict) -> float:
    """Euclidean distance between two object centroids."""
    return float(np.linalg.norm(_centroid(a) - _centroid(b)))


def is_next_to(a: dict, b: dict) -> bool:
    """
    True when centroid distance < NEXT_TO_DIST AND neither is on_top_of the other.
    """
    if centroid_dist(a, b) >= NEXT_TO_DIST:
        return False
    if is_on_top_of(a, b) or is_on_top_of(b, a):
        return False
    return True


def is_near_wall(node: dict,
                 s_min: np.ndarray,
                 s_max: np.ndarray,
                 all_nodes: list[dict] | None = None) -> bool:
    """
    True in two cases:

    1. The node itself has a WALL_LABELS label (door, window, shelf, wall) —
       architectural elements that are by definition at or part of a wall.

    2. The node centroid is within WALL_PROXIMITY_DIST of any wall-label node.

    Why not use the scene bbox faces?
    In this dataset the 3D-lifted bboxes are noisy (one bbox spans the full
    room width), so the centroid-based scene bbox is only ~2 m across and
    every object sits near its faces.  Using architectural-element proximity
    is more robust and semantically meaningful for a robot planner.
    """
    label = node.get("label", "").lower()

    # Case 1 — the node IS an architectural wall feature
    if label in WALL_LABELS:
        return True

    # Case 2 — the node is close to a wall-label node
    if all_nodes:
        c = _centroid(node)
        for other in all_nodes:
            if other["id"] == node["id"]:
                continue
            if other.get("label", "").lower() in WALL_LABELS:
                dist = float(np.linalg.norm(c - _centroid(other)))
                if dist <= WALL_PROXIMITY_DIST:
                    return True

    return False


def is_between(a: dict, b: dict, c_node: dict) -> bool:
    """
    True when A's XZ centroid lies on the line segment between
    B's and C's XZ centroids within ± BETWEEN_MARGIN metres.

    Algorithm:
      - Parameterise the B→C line segment: P(t) = B_xz + t*(C_xz - B_xz), t ∈ [0,1]
      - Find t* = argmin ||A_xz - P(t)||
      - Accept if 0 ≤ t* ≤ 1 AND perpendicular distance ≤ BETWEEN_MARGIN
    """
    a_xz = _centroid(a)[[0, 2]]
    b_xz = _centroid(b)[[0, 2]]
    c_xz = _centroid(c_node)[[0, 2]]

    bc = c_xz - b_xz
    bc_len_sq = float(np.dot(bc, bc))

    if bc_len_sq < 1e-9:
        return False   # B and C are at the same XZ position

    t_star = float(np.dot(a_xz - b_xz, bc)) / bc_len_sq
    if not (0.0 <= t_star <= 1.0):
        return False   # A's projection falls outside segment

    closest = b_xz + t_star * bc
    perp_dist = float(np.linalg.norm(a_xz - closest))

    return perp_dist <= BETWEEN_MARGIN


# ─────────────────────────────────────────────────────────────────────────────
# Edge computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_edges(nodes: list[dict],
                  s_min: np.ndarray,
                  s_max: np.ndarray) -> list[dict]:
    """
    Compute all spatial edges for the scene graph.

    Returns a list of edge dicts:
        {"source": id_A, "target": id_B, "relation": str, "distance_m": float}

    Relations: on_top_of | next_to | near_wall | between
    """
    edges: list[dict] = []

    # -- near_wall (unary, one node at a time) --------------------------------
    # Pass all_nodes so proximity to wall-label objects can be checked.
    for node in nodes:
        if is_near_wall(node, s_min, s_max, all_nodes=nodes):
            label = node.get("label", "").lower()

            # Find the closest wall-label node (or "room boundary" if self IS wall)
            if label in WALL_LABELS:
                nearest_wall_label = label
                nearest_wall_dist  = 0.0
            else:
                nearest_wall_label = "unknown"
                nearest_wall_dist  = float("inf")
                c = _centroid(node)
                for other in nodes:
                    if other["id"] == node["id"]:
                        continue
                    if other.get("label", "").lower() in WALL_LABELS:
                        d = float(np.linalg.norm(c - _centroid(other)))
                        if d < nearest_wall_dist:
                            nearest_wall_dist  = d
                            nearest_wall_label = other["label"]

            edges.append({
                "source":           node["id"],
                "target":           "wall",
                "relation":         "near_wall",
                "nearest_wall_obj": nearest_wall_label,
                "distance_m":       round(nearest_wall_dist, 4),
            })

    # -- pairwise relations ---------------------------------------------------
    for a, b in combinations(nodes, 2):
        dist = round(centroid_dist(a, b), 4)

        # on_top_of  (directed: A on top of B)
        if is_on_top_of(a, b):
            edges.append({
                "source":     a["id"],
                "target":     b["id"],
                "relation":   "on_top_of",
                "distance_m": dist,
            })
        elif is_on_top_of(b, a):
            edges.append({
                "source":     b["id"],
                "target":     a["id"],
                "relation":   "on_top_of",
                "distance_m": dist,
            })

        # next_to  (undirected — emit once with source < target lexicographically)
        if is_next_to(a, b):
            src, tgt = sorted([a["id"], b["id"]])
            edges.append({
                "source":     src,
                "target":     tgt,
                "relation":   "next_to",
                "distance_m": dist,
            })

    # -- between  (ternary: A between B and C) --------------------------------
    for idx_a, node_a in enumerate(nodes):
        # Choose every pair (B, C) that does NOT include A
        other_nodes = [n for n in nodes if n["id"] != node_a["id"]]
        for node_b, node_c in combinations(other_nodes, 2):
            if is_between(node_a, node_b, node_c):
                edges.append({
                    "source":     node_a["id"],
                    "target_1":   node_b["id"],
                    "target_2":   node_c["id"],
                    "relation":   "between",
                    "distance_m": round(
                        (centroid_dist(node_a, node_b) +
                         centroid_dist(node_a, node_c)) / 2.0, 4
                    ),
                })

    return edges


# ─────────────────────────────────────────────────────────────────────────────
# Room summary
# ─────────────────────────────────────────────────────────────────────────────

def build_room_summary(nodes: list[dict],
                       s_min: np.ndarray,
                       s_max: np.ndarray,
                       metadata: dict) -> dict:
    """Build the room_summary section of the scene graph."""
    dims = s_max - s_min
    w, h, d = round(float(dims[0]), 3), round(float(dims[1]), 3), round(float(dims[2]), 3)

    n_sparse   = sum(1 for n in nodes if n["provenance"] == "sparse")
    n_inferred = sum(1 for n in nodes if n["provenance"] == "inferred")
    nav_pct    = float(metadata.get("pct_medium", 0.0))   # medium = navigable band

    return {
        "room_dimensions_m":        [w, h, d],
        "num_objects":              len(nodes),
        "num_sparse_objects":       n_sparse,
        "num_inferred_objects":     n_inferred,
        "navigability_coverage_pct": round(nav_pct, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Human-readable description
# ─────────────────────────────────────────────────────────────────────────────

_CONF_LABEL = {
    "observed": "well-observed (conf > 0.7)",
    "sparse":   "partially observed (conf 0.3–0.7)",
    "inferred": "uncertain / inferred (conf < 0.3)",
}

_PROVENANCE_SYMBOL = {
    "observed": "●",
    "sparse":   "◆",
    "inferred": "▲",
}


def print_room_description(nodes: list[dict],
                           edges: list[dict],
                           room_summary: dict) -> None:
    """Print a human-readable room description to stdout."""
    w, h, d = room_summary["room_dimensions_m"]
    nav_pct = room_summary["navigability_coverage_pct"]

    print()
    print("╔" + "═" * 62 + "╗")
    print("║          RoboScene+ — Room Description                       ║")
    print("╚" + "═" * 62 + "╝")
    print(f"\n  Room dimensions : {w:.2f} m (W) × {h:.2f} m (H) × {d:.2f} m (D)")
    print(f"  Navigable area  : {nav_pct:.1f}% of voxels in medium-confidence band")
    print(f"  Objects tracked : {room_summary['num_objects']}")
    print(f"  Provenance split: "
          f"{room_summary['num_objects'] - room_summary['num_sparse_objects'] - room_summary['num_inferred_objects']} observed  "
          f"/ {room_summary['num_sparse_objects']} sparse  "
          f"/ {room_summary['num_inferred_objects']} inferred")

    print("\n  Objects (sorted by confidence):")
    print("  " + "─" * 60)
    header = f"  {'Label':<14}  {'Conf':>6}  {'Provenance':<10}  Position (X, Y, Z)"
    print(header)
    print("  " + "─" * 60)

    for n in sorted(nodes, key=lambda x: x["reconstruction_confidence"], reverse=True):
        sym  = _PROVENANCE_SYMBOL.get(n["provenance"], "?")
        conf = n["reconstruction_confidence"]
        x, y, z = n["position_3d"]
        print(f"  {sym} {n['label']:<12}  {conf:>6.3f}  "
              f"{n['provenance']:<10}  ({x:.2f}m, {y:.2f}m, {z:.2f}m)")

    print("  " + "─" * 60)

    # Group edges by relation for readability
    relation_groups: dict[str, list[dict]] = {}
    for e in edges:
        relation_groups.setdefault(e["relation"], []).append(e)

    if relation_groups:
        print("\n  Spatial relations:")
        for rel, rel_edges in sorted(relation_groups.items()):
            print(f"\n    [{rel.upper()}]")
            for e in rel_edges:
                if rel == "between":
                    print(f"      {e['source']}  ←between→  "
                          f"{e['target_1']} & {e['target_2']}"
                          f"  (avg dist {e['distance_m']:.2f} m)")
                elif rel == "near_wall":
                    wall_ref = e.get('nearest_wall_obj', 'wall')
                    dist_str = f"  ({e['distance_m']:.2f} m)" if e['distance_m'] > 0 else ""
                    print(f"      {e['source']}  →  near {wall_ref}{dist_str}")
                else:
                    print(f"      {e['source']}  →  {e['target']}"
                          f"  ({e['distance_m']:.2f} m)")

    print()
    print("  Legend:  ● observed  ◆ sparse  ▲ inferred")
    print("  Confidence interpretation:")
    for k, v in _CONF_LABEL.items():
        print(f"    {_PROVENANCE_SYMBOL[k]}  {v}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_scene_graph(objects_file: str = "outputs/objects_3d.json",
                      metadata_file: str = "outputs/confidence_metadata.json",
                      output_path: str   = "outputs/scene_graph.json") -> dict:
    """
    Build and save the scene graph.  Returns the graph dict.
    Can be called programmatically from other modules (e.g. app.py).
    """
    # ── 1. Load data ─────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading objects from {objects_file} …")
    if not Path(objects_file).exists():
        sys.exit(f"ERROR: {objects_file} not found.  "
                 "Run compute_confidence.py first.")

    objects    = load_objects(objects_file)
    metadata   = load_confidence_metadata(metadata_file)

    print(f"      Loaded {len(objects)} objects.")
    if metadata:
        print(f"      Confidence metadata: voxel_size={metadata.get('voxel_size')} m  "
              f"navigable={metadata.get('pct_medium', 0):.1f}%")

    # ── 2. Build nodes ────────────────────────────────────────────────────────
    print("[2/5] Building scene graph nodes …")
    nodes = [make_node(obj) for obj in objects]
    print(f"      {len(nodes)} nodes created.")

    # ── 3. Scene bounding box ────────────────────────────────────────────────
    print("[3/5] Computing scene bounding box …")
    # Physical room extent from all bbox corners (for room_dimensions_m)
    s_min, s_max = scene_bbox(nodes)
    dims = s_max - s_min
    print(f"      Bbox union   min={s_min.round(3)}  max={s_max.round(3)}")
    print(f"      Room dims    {dims[0]:.2f} m × {dims[1]:.2f} m × {dims[2]:.2f} m")

    # Centroid cloud for proximity-based edge predicates (tighter, avoids
    # noisy open-set bbox spans dominating distance calculations)
    c_min, c_max = centroid_cloud_bbox(nodes)
    print(f"      Centroid min={c_min.round(3)}  max={c_max.round(3)}")

    # ── 4. Compute edges ─────────────────────────────────────────────────────
    print("[4/5] Computing spatial edges …")
    edges = compute_edges(nodes, c_min, c_max)

    # Tally by relation type
    rel_counts: dict[str, int] = {}
    for e in edges:
        rel_counts[e["relation"]] = rel_counts.get(e["relation"], 0) + 1
    for rel, cnt in sorted(rel_counts.items()):
        print(f"      {rel:<14} {cnt:>3} edges")
    print(f"      Total       {len(edges):>3} edges")

    # ── 5. Room summary ──────────────────────────────────────────────────────
    print("[5/5] Building room summary …")
    room_summary = build_room_summary(nodes, s_min, s_max, metadata)

    # ── Assemble graph ───────────────────────────────────────────────────────
    graph = {
        "nodes":        nodes,
        "edges":        edges,
        "room_summary": room_summary,
    }

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"\n  ✓  Scene graph saved → {out_path}")
    print(f"     {len(nodes)} nodes  |  {len(edges)} edges")

    # ── Human-readable description ───────────────────────────────────────────
    print_room_description(nodes, edges, room_summary)

    return graph


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a spatial scene graph from objects_3d.json."
    )
    p.add_argument("--objects_file",  default="outputs/objects_3d.json",
                   help="Path to objects_3d.json (Session 6 output)")
    p.add_argument("--metadata_file", default="outputs/confidence_metadata.json",
                   help="Path to confidence_metadata.json")
    p.add_argument("--output",        default="outputs/scene_graph.json",
                   help="Output path for scene_graph.json")
    return p.parse_args()


def main():
    args = parse_args()
    build_scene_graph(
        objects_file  = args.objects_file,
        metadata_file = args.metadata_file,
        output_path   = args.output,
    )


if __name__ == "__main__":
    main()