from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Optional

from flask import Blueprint, jsonify, request


graph_bp = Blueprint("graph_api", __name__)

_graph_builder: Any = None
_risk_engine: Any = None
_session_factory: Any = None
_logger = logging.getLogger(__name__)


def init_graph_api(
    graph_builder: Any,
    risk_engine: Any,
    session_factory: Any,
    logger: Optional[logging.Logger] = None,
) -> None:
    global _graph_builder, _risk_engine, _session_factory, _logger
    _graph_builder = graph_builder
    _risk_engine = risk_engine
    _session_factory = session_factory
    if logger is not None:
        _logger = logger


def _require_graph_services() -> Optional[Any]:
    if _graph_builder is None or _risk_engine is None:
        return jsonify({"error": "Graph services are not initialized"}), 503
    return None


def _as_int(name: str, default: int, min_value: int, max_value: int) -> int:
    try:
        value = int(request.args.get(name, default))
    except Exception:
        value = default
    return max(min_value, min(max_value, value))


def _as_float(name: str, default: float, min_value: float, max_value: float) -> float:
    try:
        value = float(request.args.get(name, default))
    except Exception:
        value = default
    return max(min_value, min(max_value, value))


def _as_bool(name: str, default: bool = False) -> bool:
    raw = request.args.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _to_datetime(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    try:
        epoch = float(raw)
        if epoch > 1_000_000_000_000:
            epoch = epoch / 1000.0
        return datetime.fromtimestamp(epoch)
    except Exception:
        pass

    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


@graph_bp.route("/graph/wallet/<address>", methods=["GET"])
def get_wallet_graph(address: str):
    dependency_error = _require_graph_services()
    if dependency_error:
        return dependency_error

    depth = _as_int("depth", default=2, min_value=1, max_value=4)
    limit_nodes = _as_int("limit_nodes", default=300, min_value=50, max_value=1000)
    min_risk = _as_float("min_risk", default=0.0, min_value=0.0, max_value=1.0)
    suspicious_only = _as_bool("suspicious_only", default=False)
    include_trace = _as_bool("include_trace", default=True)

    start_time = _to_datetime(request.args.get("from_ts"))
    end_time = _to_datetime(request.args.get("to_ts"))

    try:
        payload = _graph_builder.get_wallet_subgraph(
            wallet=address,
            depth=depth,
            min_risk=min_risk,
            start_time=start_time,
            end_time=end_time,
            limit_nodes=limit_nodes,
            suspicious_only=suspicious_only,
        )

        if include_trace:
            payload["propagation"] = _risk_engine.trace_risk_flow(address, depth=depth)

        return jsonify(payload)
    except Exception as exc:
        _logger.exception("Graph wallet lookup failed for %s: %s", address, exc)
        return jsonify({"error": "Failed to build wallet graph"}), 500


@graph_bp.route("/graph/trace/<address>", methods=["GET"])
def trace_wallet(address: str):
    dependency_error = _require_graph_services()
    if dependency_error:
        return dependency_error

    depth = _as_int("depth", default=3, min_value=1, max_value=6)
    min_risk = _as_float("min_risk", default=0.0, min_value=0.0, max_value=1.0)
    limit_nodes = _as_int("limit_nodes", default=250, min_value=50, max_value=1000)

    start_time = _to_datetime(request.args.get("from_ts"))
    end_time = _to_datetime(request.args.get("to_ts"))

    try:
        chain = _graph_builder.get_transaction_chain(
            wallet=address,
            depth=depth,
            min_risk=min_risk,
            start_time=start_time,
            end_time=end_time,
            limit_nodes=limit_nodes,
            suspicious_only=_as_bool("suspicious_only", default=False),
        )
        propagation = _risk_engine.trace_risk_flow(address, depth=depth)
        return jsonify(
            {
                "wallet": _graph_builder._normalize_wallet(address),
                "depth": depth,
                "trace": chain,
                "propagation": propagation,
            }
        )
    except Exception as exc:
        _logger.exception("Graph trace failed for %s: %s", address, exc)
        return jsonify({"error": "Failed to compute wallet trace"}), 500


@graph_bp.route("/graph/path", methods=["GET"])
def get_path():
    dependency_error = _require_graph_services()
    if dependency_error:
        return dependency_error

    source = request.args.get("from")
    target = request.args.get("to")
    if not source or not target:
        return jsonify({"error": "Both 'from' and 'to' query params are required"}), 400

    max_depth = _as_int("max_depth", default=8, min_value=1, max_value=12)
    directed = _as_bool("directed", default=True)

    try:
        path = _graph_builder.find_shortest_path(
            source_wallet=source,
            target_wallet=target,
            max_depth=max_depth,
            directed=directed,
        )
        if not path and directed:
            # Fallback for analysts who want relationship reachability even if direction is unavailable.
            path = _graph_builder.find_shortest_path(
                source_wallet=source,
                target_wallet=target,
                max_depth=max_depth,
                directed=False,
            )

        if not path:
            return jsonify({"path": [], "nodes": [], "edges": [], "message": "No path found"}), 404

        nodes = [
            {
                "id": f"wallet:{address}",
                "type": "wallet",
                "address": address,
                "graph_risk_score": _graph_builder.get_wallet_graph_risk(address),
                "temporal_score": _graph_builder.get_wallet_temporal_score(address),
            }
            for address in path
        ]
        edges = _graph_builder.get_path_edges(path)

        return jsonify(
            {
                "path": path,
                "hops": max(0, len(path) - 1),
                "nodes": nodes,
                "edges": edges,
            }
        )
    except Exception as exc:
        _logger.exception("Path lookup failed from %s to %s: %s", source, target, exc)
        return jsonify({"error": "Failed to compute shortest path"}), 500


@graph_bp.route("/graph/high-risk", methods=["GET"])
def get_high_risk_nodes():
    dependency_error = _require_graph_services()
    if dependency_error:
        return dependency_error

    limit = _as_int("limit", default=20, min_value=1, max_value=200)
    min_score = _as_float("min_score", default=0.6, min_value=0.0, max_value=1.0)

    try:
        nodes = _risk_engine.get_top_risky_nodes(limit=limit, min_score=min_score)
        return jsonify(
            {
                "limit": limit,
                "min_score": min_score,
                "count": len(nodes),
                "nodes": nodes,
            }
        )
    except Exception as exc:
        _logger.exception("High-risk node query failed: %s", exc)
        return jsonify({"error": "Failed to fetch high-risk nodes"}), 500
