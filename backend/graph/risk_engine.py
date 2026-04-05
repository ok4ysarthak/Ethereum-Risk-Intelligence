from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple


class RiskPropagationEngine:
    """Propagate risk through wallet transfer graph with hop-decay and loop protection."""

    def __init__(
        self,
        graph_builder: Any,
        alpha: float = 0.65,
        hop_decay: float = 0.55,
        max_depth: int = 3,
    ) -> None:
        self.graph_builder = graph_builder
        self.alpha = max(0.05, min(float(alpha), 2.0))
        self.hop_decay = max(0.05, min(float(hop_decay), 1.0))
        self.max_depth = max(1, min(int(max_depth), 6))

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))

    def _normalize_wallet(self, wallet: Optional[str]) -> Optional[str]:
        return self.graph_builder._normalize_wallet(wallet)  # Uses same canonicalization logic.

    def propagate_from_sources(
        self,
        source_wallets: Iterable[str],
        depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Multi-hop risk spread:
        propagated_risk = alpha * source_risk * hop_decay^(distance-1) / (1 + distance)
        final_risk = base_risk + sum(propagated_risk)
        """
        effective_depth = self.max_depth if depth is None else max(1, min(int(depth), 6))

        normalized_sources: List[str] = []
        seen_sources: Set[str] = set()
        for raw in source_wallets or []:
            normalized = self._normalize_wallet(raw)
            if not normalized or normalized in seen_sources:
                continue
            seen_sources.add(normalized)
            normalized_sources.append(normalized)

        contributions: Dict[str, float] = defaultdict(float)
        touched_wallets: Set[str] = set()
        source_trace: Dict[str, Dict[str, Any]] = {}

        for source in normalized_sources:
            source_base_risk = float(self.graph_builder.get_wallet_base_risk(source))
            if source_base_risk <= 0.0:
                continue

            touched_wallets.add(source)
            queue: Deque[Tuple[str, int]] = deque([(source, 0)])
            visited: Set[str] = {source}
            traversed_edges = 0

            while queue:
                current_wallet, current_depth = queue.popleft()
                if current_depth >= effective_depth:
                    continue

                neighbors = self.graph_builder.get_out_neighbors(current_wallet)
                for neighbor in neighbors.keys():
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_depth = current_depth + 1
                    traversed_edges += 1

                    propagated = (
                        self.alpha
                        * source_base_risk
                        * (self.hop_decay ** max(0, next_depth - 1))
                        / (1.0 + next_depth)
                    )
                    contributions[neighbor] += propagated
                    touched_wallets.add(neighbor)
                    queue.append((neighbor, next_depth))

            source_trace[source] = {
                "source_base_risk": round(source_base_risk, 6),
                "visited_wallets": len(visited),
                "traversed_edges": traversed_edges,
            }

        final_scores: Dict[str, float] = {}
        for wallet in touched_wallets:
            base_risk = float(self.graph_builder.get_wallet_base_risk(wallet))
            propagated_risk = float(contributions.get(wallet, 0.0))
            final_score = self._clamp(base_risk + propagated_risk)
            final_scores[wallet] = round(final_score, 6)

        for source in normalized_sources:
            if source not in final_scores:
                final_scores[source] = round(self._clamp(float(self.graph_builder.get_wallet_base_risk(source))), 6)

        self.graph_builder.update_graph_risk_scores(final_scores)

        return {
            "sources": normalized_sources,
            "depth": effective_depth,
            "scores": final_scores,
            "source_trace": source_trace,
        }

    def _persist_wallet_graph_scores(
        self,
        session: Any,
        score_map: Dict[str, float],
        tx_timestamp: Optional[Any] = None,
    ) -> None:
        if session is None or not score_map:
            return

        from database.models_db import Wallet

        addresses = list(score_map.keys())
        timestamp = self.graph_builder._to_datetime(tx_timestamp)
        existing_rows = session.query(Wallet).filter(Wallet.address.in_(addresses)).all()
        wallet_rows = {row.address: row for row in existing_rows}

        for address, graph_risk in score_map.items():
            row = wallet_rows.get(address)
            if row is None:
                row = Wallet(
                    address=address,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    trust_score=max(0.0, float(self.graph_builder.get_wallet_temporal_score(address) or 0.0)),
                    avg_risk=float(self.graph_builder.get_wallet_base_risk(address) or 0.0),
                    labels={},
                    metadata_json={},
                )
                session.add(row)
                wallet_rows[address] = row

            if row.last_seen is None or timestamp > row.last_seen:
                row.last_seen = timestamp

            metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
            wallet_shap = None
            try:
                wallet_shap = self.graph_builder.get_wallet_shap(address)
            except Exception:
                wallet_shap = None
            metadata.update(
                {
                    "graph_risk_score": round(float(graph_risk), 6),
                    "graph_base_risk": round(float(self.graph_builder.get_wallet_base_risk(address)), 6),
                    "graph_neighbor_count": int(self.graph_builder.get_neighbor_count(address)),
                    "graph_updated_at": datetime.utcnow().isoformat(),
                    "graph_alerts": self.graph_builder.get_wallet_alerts(address, limit=10),
                }
            )
            if wallet_shap is not None:
                metadata["wallet_shap_explanation"] = wallet_shap
            row.metadata_json = metadata

            labels = row.labels if isinstance(row.labels, dict) else {}
            labels["graph_high_risk"] = bool(graph_risk >= 0.7)
            labels["graph_medium_risk"] = bool(0.4 <= graph_risk < 0.7)
            row.labels = labels

            if row.avg_risk is None:
                row.avg_risk = float(self.graph_builder.get_wallet_base_risk(address) or 0.0)

            if row.trust_score is None:
                row.trust_score = float(self.graph_builder.get_wallet_temporal_score(address) or 0.0)

            session.add(row)

    def update_after_transaction(
        self,
        tx_payload: Dict[str, Any],
        session: Any = None,
        depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        source_wallets = [
            tx_payload.get("from")
            or tx_payload.get("from_address")
            or tx_payload.get("wallet_address"),
            tx_payload.get("to") or tx_payload.get("to_address"),
        ]

        propagation = self.propagate_from_sources(source_wallets=source_wallets, depth=depth)
        self._persist_wallet_graph_scores(
            session=session,
            score_map=propagation.get("scores", {}),
            tx_timestamp=tx_payload.get("timestamp"),
        )

        return propagation

    def trace_risk_flow(
        self,
        target_wallet: str,
        depth: Optional[int] = None,
        max_sources: int = 20,
    ) -> List[Dict[str, Any]]:
        target = self._normalize_wallet(target_wallet)
        if not target:
            return []

        effective_depth = self.max_depth if depth is None else max(1, min(int(depth), 6))
        candidates = self.graph_builder.get_high_risk_wallets(
            limit=max(5, min(int(max_sources), 100)),
            min_score=0.4,
        )

        traces: List[Dict[str, Any]] = []
        for candidate in candidates:
            source = candidate.get("address")
            if not source or source == target:
                continue

            path = self.graph_builder.find_shortest_path(
                source_wallet=source,
                target_wallet=target,
                max_depth=effective_depth,
                directed=True,
            )
            if not path or len(path) < 2:
                continue

            distance = len(path) - 1
            source_risk = float(candidate.get("graph_risk_score") or candidate.get("base_risk_score") or 0.0)
            contribution = (
                self.alpha * source_risk * (self.hop_decay ** max(0, distance - 1)) / (1.0 + distance)
            )
            traces.append(
                {
                    "source": source,
                    "target": target,
                    "distance": distance,
                    "path": path,
                    "source_risk": round(source_risk, 6),
                    "estimated_contribution": round(self._clamp(contribution), 6),
                }
            )

        traces.sort(key=lambda item: item["estimated_contribution"], reverse=True)
        return traces[:20]

    def get_top_risky_nodes(self, limit: int = 20, min_score: float = 0.6) -> List[Dict[str, Any]]:
        nodes = self.graph_builder.get_high_risk_wallets(limit=limit, min_score=min_score)
        if nodes:
            return nodes

        # If graph_risk has not been propagated yet, seed it from highest base-risk wallets.
        seeds = self.graph_builder.get_base_risk_wallets(limit=max(10, limit), min_score=min_score)
        if not seeds:
            return []

        seed_wallets = [row["address"] for row in seeds]
        self.propagate_from_sources(seed_wallets, depth=self.max_depth)
        return self.graph_builder.get_high_risk_wallets(limit=limit, min_score=min_score)
