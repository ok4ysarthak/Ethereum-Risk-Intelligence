from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta
import math
import re
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Set, Tuple


TX_HASH_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")
PLACEHOLDER_TX_HASH_RE = re.compile(r"^0x([a-fA-F0-9])\1{63}$")


class GraphBuilder:
    """In-memory graph index for wallet/transaction relationships with DB edge persistence."""

    def __init__(
        self,
        funding_threshold: int = 3,
        related_threshold: int = 5,
        burst_window_seconds: int = 300,
        burst_threshold: int = 8,
        cache_ttl_seconds: int = 20,
        max_transactions_per_wallet: int = 1000,
        max_total_transactions: int = 30000,
    ) -> None:
        self._lock = threading.RLock()

        self.funding_threshold = max(2, int(funding_threshold))
        self.related_threshold = max(3, int(related_threshold))
        self.burst_window_seconds = max(60, int(burst_window_seconds))
        self.burst_threshold = max(4, int(burst_threshold))
        self.cache_ttl_seconds = max(5, int(cache_ttl_seconds))
        self.max_transactions_per_wallet = max(200, int(max_transactions_per_wallet))
        self.max_total_transactions = max(2000, int(max_total_transactions))

        self._transactions: Dict[str, Dict[str, Any]] = {}
        self._tx_to_edges: Dict[str, List[Dict[str, Any]]] = {}
        self._transaction_order: Deque[str] = deque()

        self._wallet_transactions: Dict[str, Deque[str]] = defaultdict(deque)
        self._wallet_activity: Dict[str, Deque[datetime]] = defaultdict(deque)

        self._wallet_outgoing: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._wallet_incoming: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._wallet_neighbors: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        self._pair_history: Dict[Tuple[str, str], Deque[datetime]] = defaultdict(deque)
        self._wallet_alerts: Dict[str, Deque[Dict[str, Any]]] = defaultdict(deque)

        self._wallet_base_risk: Dict[str, float] = defaultdict(float)
        self._wallet_graph_risk: Dict[str, float] = defaultdict(float)
        self._wallet_temporal_score: Dict[str, float] = defaultdict(float)
        self._wallet_last_shap: Dict[str, Any] = {}

        self._cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._version = 0

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))

    @staticmethod
    def _normalize_wallet(address: Optional[str]) -> Optional[str]:
        if not address:
            return None
        value = str(address).strip().lower()
        if not value:
            return None
        if not value.startswith("0x"):
            value = f"0x{value}"
        return value

    @staticmethod
    def _normalize_tx_hash(tx_hash: Optional[str]) -> Optional[str]:
        if not tx_hash:
            return None
        value = str(tx_hash).strip().lower()
        if not value:
            return None
        if not value.startswith("0x"):
            value = f"0x{value}"
        if not TX_HASH_RE.match(value):
            return None
        if PLACEHOLDER_TX_HASH_RE.match(value):
            return None
        return value

    @staticmethod
    def _to_datetime(value: Any) -> datetime:
        if value is None:
            return datetime.utcnow()

        if isinstance(value, datetime):
            return value

        try:
            raw = float(value)
            if raw > 1_000_000_000_000:
                raw = raw / 1000.0
            return datetime.fromtimestamp(raw)
        except Exception:
            pass

        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return datetime.utcnow()

    @staticmethod
    def _to_epoch_seconds(value: Optional[datetime]) -> Optional[int]:
        if value is None:
            return None
        return int(value.timestamp())

    def _parse_risk_probability(self, payload: Dict[str, Any]) -> float:
        candidates = [
            payload.get("fraud_probability"),
            payload.get("risk_probability"),
            payload.get("riskScore"),
            payload.get("risk_score"),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                parsed = float(candidate)
                if parsed > 1.0:
                    parsed = parsed / 10.0
                return self._clamp(parsed)
            except Exception:
                continue
        return 0.0

    def _parse_temporal_score(self, payload: Dict[str, Any]) -> float:
        candidates = [
            payload.get("temporal_score_normalized"),
            payload.get("wallet_trust_score"),
            payload.get("walletScore"),
            payload.get("temporal_score"),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                parsed = float(candidate)
                if parsed > 1.0:
                    parsed = parsed / 10.0
                return self._clamp(parsed)
            except Exception:
                continue
        return 0.0

    @staticmethod
    def _coerce_explanation_payload(value: Any) -> Optional[Any]:
        if value is None:
            return None
        if isinstance(value, (dict, list, str)):
            return value
        try:
            return str(value)
        except Exception:
            return None

    def _parse_transaction_shap(self, payload: Dict[str, Any]) -> Optional[Any]:
        explanation = payload.get("transaction_shap_explanation")
        if explanation is None:
            explanation = payload.get("shap_explanation")
        coerced = self._coerce_explanation_payload(explanation)
        if isinstance(coerced, dict):
            if str(coerced.get("kind") or "").strip().lower() == "wallet":
                return None
        return coerced

    def _build_transaction_shap_explanation(
        self,
        payload: Dict[str, Any],
        risk_probability: float,
        value_eth: float,
    ) -> Dict[str, Any]:
        explicit = self._parse_transaction_shap(payload)
        if explicit is not None:
            if isinstance(explicit, dict):
                cleaned = dict(explicit)
                cleaned["kind"] = "transaction"
                cleaned.setdefault(
                    "summary",
                    "Transaction-level SHAP explanation generated from transaction model features.",
                )
                if not isinstance(cleaned.get("top_features"), list):
                    cleaned["top_features"] = []
                return cleaned
            if isinstance(explicit, list):
                return {
                    "kind": "transaction",
                    "summary": "Transaction-level SHAP factors captured from model output.",
                    "top_features": explicit[:6],
                }
            return {
                "kind": "transaction",
                "summary": str(explicit),
                "top_features": [],
            }

        factors: List[Dict[str, Any]] = []

        def add_factor(feature: str, meaning: str, impact: float, direction: str) -> None:
            factors.append(
                {
                    "feature": feature,
                    "meaning": meaning,
                    "impact": f"{impact:+.4f}",
                    "absolute_impact": round(abs(float(impact)), 4),
                    "direction": direction,
                }
            )

        tx_value = max(0.0, float(value_eth or 0.0))
        risk_component = max(0.0, min(1.0, float(risk_probability)))
        add_factor(
            "Fraud_Probability",
            "Model-estimated probability of fraudulent transaction behavior",
            risk_component,
            "increases_risk" if risk_component >= 0.5 else "reduces_risk",
        )

        add_factor(
            "Transaction_Value",
            "Transferred ETH value contribution",
            min(0.45, math.log1p(tx_value) / 9.0),
            "increases_risk" if tx_value > 2.0 else "reduces_risk",
        )

        tx_velocity = float(payload.get("Transaction_Velocity") or 0.0)
        if tx_velocity > 0:
            add_factor(
                "Transaction_Velocity",
                "Short-interval transfer velocity",
                min(0.4, tx_velocity / 30.0),
                "increases_risk",
            )

        gas_price = float(payload.get("Gas_Price") or 0.0)
        if gas_price > 0:
            add_factor(
                "Gas_Price",
                "Gas urgency and bidding pressure",
                min(0.3, math.log1p(gas_price) / 20.0),
                "increases_risk",
            )

        tx_fees = float(payload.get("Transaction_Fees") or 0.0)
        if tx_fees > 0:
            add_factor(
                "Transaction_Fees",
                "Observed fee profile for execution",
                min(0.28, math.log1p(tx_fees) / 12.0),
                "increases_risk",
            )

        factors.sort(key=lambda row: row["absolute_impact"], reverse=True)
        top_factors = factors[:5]

        summary = (
            f"Transaction explanation is derived from transaction-centric model signals "
            f"(risk={risk_probability:.2f}, value={tx_value:.4f} ETH)."
        )

        return {
            "kind": "transaction",
            "summary": summary,
            "top_features": top_factors,
        }

    def _build_wallet_shap_explanation(
        self,
        payload: Dict[str, Any],
        risk_probability: float,
        temporal_score: float,
    ) -> Optional[Any]:
        explicit = self._coerce_explanation_payload(payload.get("wallet_shap_explanation"))
        if explicit is not None:
            if isinstance(explicit, dict):
                if str(explicit.get("kind") or "").strip().lower() == "transaction":
                    explicit = None
                else:
                    cleaned = dict(explicit)
                    cleaned["kind"] = "wallet"
                    cleaned.setdefault(
                        "summary",
                        "Wallet-level SHAP explanation derived from graph and temporal wallet signals.",
                    )
                    if not isinstance(cleaned.get("top_features"), list):
                        cleaned["top_features"] = []
                    return cleaned
            elif isinstance(explicit, list):
                return {
                    "kind": "wallet",
                    "summary": "Wallet-level SHAP factors captured from wallet context.",
                    "top_features": explicit[:6],
                }
            else:
                return {
                    "kind": "wallet",
                    "summary": str(explicit),
                    "top_features": [],
                }

        temporal_state = payload.get("temporal_state")
        if not isinstance(temporal_state, dict):
            temporal_state = {}

        factors: List[Dict[str, Any]] = []

        def add_factor(feature: str, meaning: str, impact: float, direction: str) -> None:
            factors.append(
                {
                    "feature": feature,
                    "meaning": meaning,
                    "impact": f"{impact:+.4f}",
                    "absolute_impact": round(abs(float(impact)), 4),
                    "direction": direction,
                }
            )

        risk_impact = max(0.0, min(1.0, float(risk_probability)))
        add_factor(
            "Graph_Risk_Propagation",
            "Propagated risk from connected wallet graph",
            risk_impact,
            "increases_risk" if risk_impact >= 0.5 else "reduces_risk",
        )

        trust_delta = 0.5 - max(0.0, min(1.0, float(temporal_score)))
        add_factor(
            "Temporal_Trust_Score",
            "Recent wallet trust trajectory",
            trust_delta,
            "increases_risk" if trust_delta > 0 else "reduces_risk",
        )

        tx_count_1m = float(temporal_state.get("tx_count_last_1_min") or 0.0)
        if tx_count_1m > 0:
            add_factor(
                "Tx_Burst_1m",
                "Short-window transfer burst frequency",
                min(0.45, tx_count_1m / 40.0),
                "increases_risk",
            )

        if bool(temporal_state.get("burst_detected")):
            add_factor(
                "Burst_Detected",
                "Burst behavior flag in temporal model",
                0.26,
                "increases_risk",
            )

        if bool(temporal_state.get("dormant_spike")):
            add_factor(
                "Dormant_Reactivation",
                "Dormant wallet reactivated with elevated activity",
                0.31,
                "increases_risk",
            )

        factors.sort(key=lambda row: row["absolute_impact"], reverse=True)
        top_factors = factors[:5]

        summary = (
            f"Wallet explanation combines graph-risk ({risk_probability:.2f}) and temporal trust ({temporal_score:.2f}) "
            "to estimate wallet-level threat exposure."
        )
        if bool(temporal_state.get("burst_detected")) or bool(temporal_state.get("dormant_spike")):
            summary += " Temporal anomaly flags are currently elevating wallet risk."

        return {
            "kind": "wallet",
            "summary": summary,
            "top_features": top_factors,
        }

    @staticmethod
    def _risk_color(risk_score: float) -> str:
        if risk_score >= 0.7:
            return "#ef4444"
        if risk_score >= 0.4:
            return "#f59e0b"
        return "#22c55e"

    def _cache_get(self, key: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
        item = self._cache.get(key)
        if not item:
            return None
        if item["version"] != self._version or item["expires_at"] < time.time():
            self._cache.pop(key, None)
            return None
        return item["value"]

    def _cache_set(self, key: Tuple[Any, ...], value: Dict[str, Any]) -> None:
        self._cache[key] = {
            "version": self._version,
            "expires_at": time.time() + self.cache_ttl_seconds,
            "value": value,
        }

    def _invalidate_cache_locked(self) -> None:
        self._version += 1
        if len(self._cache) > 500:
            self._cache.clear()

    def _remove_tx_from_wallet_index_locked(self, wallet: Optional[str], tx_hash: str) -> None:
        if not wallet:
            return
        wallet_txs = self._wallet_transactions.get(wallet)
        if not wallet_txs:
            return
        try:
            wallet_txs.remove(tx_hash)
        except ValueError:
            return
        if not wallet_txs:
            self._wallet_transactions.pop(wallet, None)

    @staticmethod
    def _decrement_relation_count_locked(
        relation_map: Dict[str, Dict[str, int]],
        source_wallet: Optional[str],
        target_wallet: Optional[str],
    ) -> None:
        if not source_wallet or not target_wallet:
            return

        bucket = relation_map.get(source_wallet)
        if not bucket:
            return

        current = int(bucket.get(target_wallet, 0) or 0)
        if current <= 1:
            bucket.pop(target_wallet, None)
        else:
            bucket[target_wallet] = current - 1

        if not bucket:
            relation_map.pop(source_wallet, None)

    def _remove_tx_relationship_indexes_locked(self, tx: Dict[str, Any]) -> None:
        from_wallet = tx.get("from_address")
        to_wallet = tx.get("to_address")
        timestamp = tx.get("timestamp")

        if from_wallet and to_wallet:
            self._decrement_relation_count_locked(self._wallet_outgoing, from_wallet, to_wallet)
            self._decrement_relation_count_locked(self._wallet_incoming, to_wallet, from_wallet)
            self._decrement_relation_count_locked(self._wallet_neighbors, from_wallet, to_wallet)
            self._decrement_relation_count_locked(self._wallet_neighbors, to_wallet, from_wallet)

            pair = (from_wallet, to_wallet)
            history = self._pair_history.get(pair)
            if history is not None and timestamp is not None:
                try:
                    history.remove(timestamp)
                except ValueError:
                    pass
                if not history:
                    self._pair_history.pop(pair, None)

        for wallet in (from_wallet, to_wallet):
            if not wallet or timestamp is None:
                continue

            activity = self._wallet_activity.get(wallet)
            if not activity:
                continue
            try:
                activity.remove(timestamp)
            except ValueError:
                pass
            if not activity:
                self._wallet_activity.pop(wallet, None)

    def _prune_global_transactions_locked(self) -> None:
        while len(self._transactions) > self.max_total_transactions and self._transaction_order:
            oldest_tx_hash = self._transaction_order.pop()
            tx = self._transactions.pop(oldest_tx_hash, None)
            self._tx_to_edges.pop(oldest_tx_hash, None)
            if not tx:
                continue

            self._remove_tx_from_wallet_index_locked(tx.get("from_address"), oldest_tx_hash)
            self._remove_tx_from_wallet_index_locked(tx.get("to_address"), oldest_tx_hash)
            self._remove_tx_relationship_indexes_locked(tx)

    def _record_alert_locked(
        self,
        wallet: str,
        alert_type: str,
        tx_hash: Optional[str],
        severity: str,
        details: Dict[str, Any],
    ) -> None:
        entry = {
            "type": alert_type,
            "tx_hash": tx_hash,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details,
        }
        queue = self._wallet_alerts[wallet]
        queue.appendleft(entry)
        while len(queue) > 60:
            queue.pop()

    def _build_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        weight: float,
        timestamp: datetime,
        tx_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tx_part = tx_hash or "none"
        edge_id = f"{source}->{target}:{edge_type}:{tx_part}:{int(timestamp.timestamp())}"
        return {
            "id": edge_id,
            "source": source,
            "target": target,
            "edge_type": edge_type,
            "weight": round(float(weight), 4),
            "timestamp": timestamp.isoformat(),
            "tx_hash": tx_hash,
            "metadata": metadata or {},
        }

    def _find_wallet_path_locked(
        self,
        source_wallet: str,
        target_wallet: str,
        max_depth: int = 5,
        directed: bool = True,
    ) -> Optional[List[str]]:
        if source_wallet == target_wallet:
            return [source_wallet]

        max_depth = max(1, int(max_depth))
        frontier: Deque[Tuple[str, List[str]]] = deque([(source_wallet, [source_wallet])])
        visited: Set[str] = {source_wallet}

        while frontier:
            current, path = frontier.popleft()
            if len(path) - 1 >= max_depth:
                continue

            if directed:
                neighbors = self._wallet_outgoing.get(current, {})
            else:
                outgoing = self._wallet_outgoing.get(current, {})
                incoming = self._wallet_incoming.get(current, {})
                neighbors = {**incoming, **outgoing}

            for neighbor in neighbors.keys():
                if neighbor in visited:
                    continue
                next_path = path + [neighbor]
                if neighbor == target_wallet:
                    return next_path
                visited.add(neighbor)
                frontier.append((neighbor, next_path))

        return None

    def _detect_relationships_locked(
        self,
        from_wallet: str,
        to_wallet: str,
        timestamp: datetime,
        tx_hash: str,
    ) -> List[Dict[str, Any]]:
        derived_edges: List[Dict[str, Any]] = []

        # Repeated directed transfer => FUNDING
        pair = (from_wallet, to_wallet)
        history = self._pair_history[pair]
        history.append(timestamp)
        cutoff = timestamp - timedelta(hours=24)
        while history and history[0] < cutoff:
            history.popleft()

        if len(history) >= self.funding_threshold:
            derived_edges.append(
                self._build_edge(
                    source=f"wallet:{from_wallet}",
                    target=f"wallet:{to_wallet}",
                    edge_type="FUNDING",
                    weight=min(6.0, 1.0 + math.log1p(len(history))),
                    timestamp=timestamp,
                    tx_hash=tx_hash,
                    metadata={"events_24h": len(history)},
                )
            )
            self._record_alert_locked(
                from_wallet,
                "FUNDING",
                tx_hash,
                "medium",
                {"target": to_wallet, "events_24h": len(history)},
            )

        # Frequent interactions => RELATED
        interactions = self._wallet_neighbors[from_wallet].get(to_wallet, 0)
        if interactions >= self.related_threshold:
            derived_edges.append(
                self._build_edge(
                    source=f"wallet:{from_wallet}",
                    target=f"wallet:{to_wallet}",
                    edge_type="RELATED",
                    weight=min(5.0, 1.0 + math.log1p(interactions)),
                    timestamp=timestamp,
                    tx_hash=tx_hash,
                    metadata={"interactions": interactions},
                )
            )

        # Circular flow detection A -> B and existing path B -> ... -> A
        cycle_path = self._find_wallet_path_locked(to_wallet, from_wallet, max_depth=3, directed=True)
        if cycle_path and len(cycle_path) >= 2:
            cycle = [from_wallet] + cycle_path
            for wallet in set(cycle):
                self._record_alert_locked(
                    wallet,
                    "CIRCULAR",
                    tx_hash,
                    "high",
                    {"cycle_path": cycle},
                )
            derived_edges.append(
                self._build_edge(
                    source=f"wallet:{from_wallet}",
                    target=f"wallet:{to_wallet}",
                    edge_type="RELATED",
                    weight=3.0,
                    timestamp=timestamp,
                    tx_hash=tx_hash,
                    metadata={"relation": "circular", "path": cycle_path},
                )
            )

        # Burst cluster detection around each wallet
        burst_cutoff = timestamp - timedelta(seconds=self.burst_window_seconds)
        for wallet in (from_wallet, to_wallet):
            activity = self._wallet_activity[wallet]
            while activity and activity[0] < burst_cutoff:
                activity.popleft()

            if len(activity) < self.burst_threshold:
                continue

            counterparties: Set[str] = set()
            for tx_id in list(self._wallet_transactions[wallet])[:250]:
                tx = self._transactions.get(tx_id)
                if not tx:
                    continue
                if tx["timestamp"] < burst_cutoff:
                    continue
                other = self._counterparty_for_wallet(tx, wallet)
                if other:
                    counterparties.add(other)

            if len(counterparties) >= max(3, self.burst_threshold // 2):
                self._record_alert_locked(
                    wallet,
                    "BURST_CLUSTER",
                    tx_hash,
                    "high",
                    {
                        "events": len(activity),
                        "counterparties": len(counterparties),
                        "window_seconds": self.burst_window_seconds,
                    },
                )

        return derived_edges

    def _persist_edges_locked(self, session: Any, edges: List[Dict[str, Any]]) -> None:
        if session is None or not edges:
            return
        from database.models_db import GraphEdge

        for edge in edges:
            row = GraphEdge(
                source_node=edge["source"],
                target_node=edge["target"],
                edge_type=edge["edge_type"],
                weight=edge["weight"],
                timestamp=self._to_datetime(edge.get("timestamp")),
                metadata_json={
                    "tx_hash": edge.get("tx_hash"),
                    **(edge.get("metadata") or {}),
                },
            )
            session.add(row)

    # ------------------------------------------------------------------
    # Public mutation APIs
    # ------------------------------------------------------------------
    def add_transaction(
        self,
        tx_payload: Dict[str, Any],
        session: Any = None,
        persist_edges: bool = True,
    ) -> Dict[str, Any]:
        """Add/update a transaction and maintain graph indexes in real time."""
        tx_payload = tx_payload or {}
        tx_hash = self._normalize_tx_hash(
            tx_payload.get("transaction_hash")
            or tx_payload.get("tx_hash")
            or tx_payload.get("id")
            or tx_payload.get("hash")
        )
        if not tx_hash:
            return {"added": False, "reason": "missing_tx_hash"}

        from_wallet = self._normalize_wallet(
            tx_payload.get("from_address") or tx_payload.get("from") or tx_payload.get("wallet_address")
        )
        to_wallet = self._normalize_wallet(tx_payload.get("to_address") or tx_payload.get("to"))
        if not from_wallet and not to_wallet:
            return {"added": False, "reason": "missing_wallet_endpoints", "tx_hash": tx_hash}

        timestamp = self._to_datetime(tx_payload.get("timestamp"))

        try:
            value_eth = float(tx_payload.get("value") or tx_payload.get("amount_eth") or tx_payload.get("Transaction_Value") or 0.0)
        except Exception:
            value_eth = 0.0

        risk_probability = self._parse_risk_probability(tx_payload)
        temporal_score = self._parse_temporal_score(tx_payload)
        transaction_shap_explanation = self._build_transaction_shap_explanation(
            tx_payload,
            risk_probability,
            value_eth,
        )
        wallet_shap_explanation = self._build_wallet_shap_explanation(
            tx_payload,
            risk_probability,
            temporal_score,
        )

        with self._lock:
            if tx_hash in self._transactions:
                existing = self._transactions[tx_hash]
                existing["risk_score"] = max(float(existing.get("risk_score", 0.0)), risk_probability)
                if temporal_score:
                    existing["temporal_score"] = temporal_score
                if transaction_shap_explanation:
                    existing["transaction_shap_explanation"] = transaction_shap_explanation
                    existing["shap_explanation"] = transaction_shap_explanation
                if wallet_shap_explanation and from_wallet:
                    self._wallet_last_shap[from_wallet] = wallet_shap_explanation
                if from_wallet:
                    self._wallet_base_risk[from_wallet] = max(
                        self._wallet_base_risk.get(from_wallet, 0.0),
                        risk_probability,
                    )
                self._invalidate_cache_locked()
                return {"added": False, "reason": "duplicate", "tx_hash": tx_hash}

            tx_entry = {
                "tx_hash": tx_hash,
                "from_address": from_wallet,
                "to_address": to_wallet,
                "value": value_eth,
                "risk_score": risk_probability,
                "temporal_score": temporal_score,
                "timestamp": timestamp,
                "transaction_shap_explanation": transaction_shap_explanation,
                "shap_explanation": transaction_shap_explanation,
            }
            self._transactions[tx_hash] = tx_entry
            self._transaction_order.appendleft(tx_hash)

            edges: List[Dict[str, Any]] = []

            for wallet in (from_wallet, to_wallet):
                if not wallet:
                    continue
                wallet_txs = self._wallet_transactions[wallet]
                wallet_txs.appendleft(tx_hash)
                while len(wallet_txs) > self.max_transactions_per_wallet:
                    wallet_txs.pop()

                activity = self._wallet_activity[wallet]
                activity.append(timestamp)
                activity_cutoff = timestamp - timedelta(hours=24)
                while activity and activity[0] < activity_cutoff:
                    activity.popleft()

            if from_wallet:
                self._wallet_base_risk[from_wallet] = max(self._wallet_base_risk.get(from_wallet, 0.0) * 0.92, risk_probability)
                self._wallet_temporal_score[from_wallet] = temporal_score or self._wallet_temporal_score.get(from_wallet, 0.0)
                if wallet_shap_explanation:
                    self._wallet_last_shap[from_wallet] = wallet_shap_explanation
                edges.append(
                    self._build_edge(
                        source=f"wallet:{from_wallet}",
                        target=f"tx:{tx_hash}",
                        edge_type="INITIATED",
                        weight=1.0,
                        timestamp=timestamp,
                        tx_hash=tx_hash,
                    )
                )

            if to_wallet:
                self._wallet_base_risk[to_wallet] = max(self._wallet_base_risk.get(to_wallet, 0.0) * 0.95, risk_probability * 0.85)
                edges.append(
                    self._build_edge(
                        source=f"tx:{tx_hash}",
                        target=f"wallet:{to_wallet}",
                        edge_type="RECEIVED",
                        weight=1.0,
                        timestamp=timestamp,
                        tx_hash=tx_hash,
                    )
                )

            if from_wallet and to_wallet:
                self._wallet_outgoing[from_wallet][to_wallet] += 1
                self._wallet_incoming[to_wallet][from_wallet] += 1

                self._wallet_neighbors[from_wallet][to_wallet] += 1
                self._wallet_neighbors[to_wallet][from_wallet] += 1

                interaction_count = self._wallet_outgoing[from_wallet][to_wallet]
                transfer_weight = min(6.0, 1.0 + math.log1p(interaction_count))
                edges.append(
                    self._build_edge(
                        source=f"wallet:{from_wallet}",
                        target=f"wallet:{to_wallet}",
                        edge_type="TRANSFER",
                        weight=transfer_weight,
                        timestamp=timestamp,
                        tx_hash=tx_hash,
                        metadata={"interaction_count": interaction_count, "value": value_eth},
                    )
                )
                edges.extend(self._detect_relationships_locked(from_wallet, to_wallet, timestamp, tx_hash))

            self._tx_to_edges[tx_hash] = edges

            if persist_edges and session is not None:
                self._persist_edges_locked(session, edges)

            self._prune_global_transactions_locked()

            self._invalidate_cache_locked()

            return {
                "added": True,
                "tx_hash": tx_hash,
                "edges_added": len(edges),
                "from": from_wallet,
                "to": to_wallet,
            }

    def bootstrap_from_database(self, session_factory: Any, limit: int = 5000) -> Dict[str, Any]:
        """Warm in-memory graph from recent persisted transactions."""
        from database.models_db import GraphEdge, Transaction

        limit = max(100, min(int(limit), 20000))
        session = session_factory()
        bootstrapped = 0
        populate_edges = False
        try:
            existing_edges = int(session.query(GraphEdge).count())
            populate_edges = existing_edges == 0

            rows = (
                session.query(Transaction)
                .filter(Transaction.risk_score.isnot(None))
                .filter(Transaction.wallet_trust_score.isnot(None))
                .order_by(Transaction.timestamp.desc())
                .limit(limit)
                .all()
            )

            for row in reversed(rows):
                payload = {
                    "transaction_hash": row.tx_hash,
                    "from": row.from_address,
                    "to": row.to_address,
                    "value": row.amount_eth,
                    "fraud_probability": row.risk_score,
                    "wallet_trust_score": row.wallet_trust_score,
                    "timestamp": row.timestamp,
                }
                if isinstance(row.raw_payload, dict):
                    payload["transaction_shap_explanation"] = (
                        row.raw_payload.get("transaction_shap_explanation")
                        or row.raw_payload.get("shap_explanation")
                    )
                    payload["shap_explanation"] = payload.get("transaction_shap_explanation")
                    payload["wallet_shap_explanation"] = row.raw_payload.get("wallet_shap_explanation")
                    payload["temporal_state"] = row.raw_payload.get("temporal_state")
                    payload["risk_score"] = row.raw_payload.get("risk_score")
                    if payload["fraud_probability"] is None:
                        payload["fraud_probability"] = row.raw_payload.get("fraud_probability")

                result = self.add_transaction(payload, session=session if populate_edges else None, persist_edges=populate_edges)
                if result.get("added"):
                    bootstrapped += 1

            if populate_edges:
                session.commit()

            return {
                "bootstrapped_transactions": bootstrapped,
                "edge_table_seeded": bool(populate_edges),
                "loaded_limit": limit,
            }
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Public query APIs
    # ------------------------------------------------------------------
    def get_out_neighbors(self, wallet: str) -> Dict[str, int]:
        wallet = self._normalize_wallet(wallet)
        if not wallet:
            return {}
        with self._lock:
            return dict(self._wallet_outgoing.get(wallet, {}))

    def get_wallet_base_risk(self, wallet: str) -> float:
        wallet = self._normalize_wallet(wallet)
        if not wallet:
            return 0.0
        with self._lock:
            return round(self._clamp(self._wallet_base_risk.get(wallet, 0.0)), 6)

    def get_wallet_graph_risk(self, wallet: str) -> float:
        wallet = self._normalize_wallet(wallet)
        if not wallet:
            return 0.0
        with self._lock:
            score = self._wallet_graph_risk.get(wallet, self._wallet_base_risk.get(wallet, 0.0))
            return round(self._clamp(score), 6)

    def get_wallet_temporal_score(self, wallet: str) -> float:
        wallet = self._normalize_wallet(wallet)
        if not wallet:
            return 0.0
        with self._lock:
            return round(self._clamp(self._wallet_temporal_score.get(wallet, 0.0)), 6)

    def update_graph_risk_scores(self, wallet_scores: Dict[str, float]) -> None:
        with self._lock:
            for wallet, score in (wallet_scores or {}).items():
                normalized = self._normalize_wallet(wallet)
                if not normalized:
                    continue
                self._wallet_graph_risk[normalized] = self._clamp(score)
            self._invalidate_cache_locked()

    def get_wallet_shap(self, wallet: str) -> Optional[Any]:
        wallet = self._normalize_wallet(wallet)
        if not wallet:
            return None
        with self._lock:
            return self._wallet_last_shap.get(wallet)

    def get_wallet_alerts(self, wallet: str, limit: int = 25) -> List[Dict[str, Any]]:
        wallet = self._normalize_wallet(wallet)
        if not wallet:
            return []
        limit = max(1, min(int(limit), 100))
        with self._lock:
            alerts = list(self._wallet_alerts.get(wallet, []))
            return alerts[:limit]

    def get_neighbor_count(self, wallet: str) -> int:
        wallet = self._normalize_wallet(wallet)
        if not wallet:
            return 0
        with self._lock:
            return len(self._wallet_neighbors.get(wallet, {}))

    def get_base_risk_wallets(self, limit: int = 25, min_score: float = 0.0) -> List[Dict[str, Any]]:
        min_score = self._clamp(min_score)
        limit = max(1, min(int(limit), 500))
        with self._lock:
            ranked = []
            for wallet, score in self._wallet_base_risk.items():
                if score < min_score:
                    continue
                ranked.append(
                    {
                        "address": wallet,
                        "base_risk_score": round(self._clamp(score), 6),
                        "graph_risk_score": round(self._clamp(self._wallet_graph_risk.get(wallet, score)), 6),
                        "temporal_score": round(self._clamp(self._wallet_temporal_score.get(wallet, 0.0)), 6),
                        "neighbor_count": len(self._wallet_neighbors.get(wallet, {})),
                    }
                )
            ranked.sort(key=lambda item: item["base_risk_score"], reverse=True)
            return ranked[:limit]

    def get_high_risk_wallets(self, limit: int = 20, min_score: float = 0.6) -> List[Dict[str, Any]]:
        min_score = self._clamp(min_score)
        limit = max(1, min(int(limit), 500))
        with self._lock:
            wallets = set(self._wallet_base_risk.keys()) | set(self._wallet_graph_risk.keys())
            ranked: List[Dict[str, Any]] = []
            for wallet in wallets:
                base_risk = self._clamp(self._wallet_base_risk.get(wallet, 0.0))
                graph_risk = self._clamp(self._wallet_graph_risk.get(wallet, base_risk))
                if graph_risk < min_score:
                    continue
                ranked.append(
                    {
                        "id": f"wallet:{wallet}",
                        "address": wallet,
                        "base_risk_score": round(base_risk, 6),
                        "graph_risk_score": round(graph_risk, 6),
                        "temporal_score": round(self._clamp(self._wallet_temporal_score.get(wallet, 0.0)), 6),
                        "neighbor_count": len(self._wallet_neighbors.get(wallet, {})),
                        "alerts": list(self._wallet_alerts.get(wallet, []))[:5],
                    }
                )
            ranked.sort(key=lambda item: item["graph_risk_score"], reverse=True)
            return ranked[:limit]

    def get_neighbors(self, wallet: str, limit: int = 50) -> List[Dict[str, Any]]:
        wallet = self._normalize_wallet(wallet)
        if not wallet:
            return []
        limit = max(1, min(int(limit), 200))

        cache_key = ("neighbors", wallet, limit)
        with self._lock:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached["neighbors"]

            outgoing = self._wallet_outgoing.get(wallet, {})
            incoming = self._wallet_incoming.get(wallet, {})
            neighbors = []
            all_addresses = set(outgoing.keys()) | set(incoming.keys())
            for neighbor in all_addresses:
                out_count = int(outgoing.get(neighbor, 0))
                in_count = int(incoming.get(neighbor, 0))
                total = out_count + in_count
                neighbors.append(
                    {
                        "address": neighbor,
                        "outgoing": out_count,
                        "incoming": in_count,
                        "interactions": total,
                        "graph_risk_score": round(self._clamp(self._wallet_graph_risk.get(neighbor, self._wallet_base_risk.get(neighbor, 0.0))), 6),
                    }
                )

            neighbors.sort(key=lambda row: row["interactions"], reverse=True)
            trimmed = neighbors[:limit]
            self._cache_set(cache_key, {"neighbors": trimmed})
            return trimmed

    def _tx_passes_filters(
        self,
        tx: Dict[str, Any],
        min_risk: float,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        suspicious_only: bool,
    ) -> bool:
        threshold = max(min_risk, 0.6 if suspicious_only else min_risk)
        if float(tx.get("risk_score", 0.0)) < threshold:
            return False

        ts = tx.get("timestamp")
        if start_time and ts < start_time:
            return False
        if end_time and ts > end_time:
            return False
        return True

    @staticmethod
    def _counterparty_for_wallet(tx: Dict[str, Any], wallet: str) -> Optional[str]:
        source = tx.get("from_address")
        target = tx.get("to_address")
        if source == wallet:
            return target
        if target == wallet:
            return source
        return None

    def _wallet_node_locked(self, wallet: str) -> Dict[str, Any]:
        graph_risk = self._clamp(self._wallet_graph_risk.get(wallet, self._wallet_base_risk.get(wallet, 0.0)))
        base_risk = self._clamp(self._wallet_base_risk.get(wallet, 0.0))
        temporal_score = self._clamp(self._wallet_temporal_score.get(wallet, 0.0))
        wallet_shap = self._wallet_last_shap.get(wallet)

        wallet_shap_summary = ""
        if isinstance(wallet_shap, dict):
            wallet_shap_summary = str(wallet_shap.get("summary") or "")
        elif isinstance(wallet_shap, str):
            wallet_shap_summary = wallet_shap

        return {
            "id": f"wallet:{wallet}",
            "label": f"{wallet[:8]}...{wallet[-6:]}",
            "type": "wallet",
            "address": wallet,
            "base_risk_score": round(base_risk, 6),
            "graph_risk_score": round(graph_risk, 6),
            "temporal_score": round(temporal_score, 6),
            "color": self._risk_color(graph_risk),
            "size": 18 if graph_risk < 0.4 else 24 if graph_risk < 0.7 else 30,
            "neighbor_count": len(self._wallet_neighbors.get(wallet, {})),
            "alerts": list(self._wallet_alerts.get(wallet, []))[:5],
            "wallet_shap_explanation": wallet_shap,
            "title": (
                f"Wallet: {wallet}<br/>"
                f"Graph risk: {graph_risk:.3f}<br/>"
                f"Base risk: {base_risk:.3f}<br/>"
                f"Temporal score: {temporal_score:.3f}"
                + (f"<br/>Wallet SHAP: {wallet_shap_summary}" if wallet_shap_summary else "")
            ),
        }

    def _tx_node_locked(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        tx_hash = tx["tx_hash"]
        risk_score = self._clamp(float(tx.get("risk_score", 0.0)))
        transaction_shap = tx.get("transaction_shap_explanation") or tx.get("shap_explanation")
        return {
            "id": f"tx:{tx_hash}",
            "label": f"tx:{tx_hash[2:10]}...",
            "type": "transaction",
            "tx_hash": tx_hash,
            "from": tx.get("from_address"),
            "to": tx.get("to_address"),
            "value": float(tx.get("value") or 0.0),
            "risk_score": round(risk_score, 6),
            "temporal_score": round(self._clamp(float(tx.get("temporal_score") or 0.0)), 6),
            "timestamp": tx["timestamp"].isoformat(),
            "color": self._risk_color(risk_score),
            "shape": "box",
            "size": 12,
            "transaction_shap_explanation": transaction_shap,
            "shap_explanation": transaction_shap,
            "title": (
                f"Transaction: {tx_hash}<br/>"
                f"Risk: {risk_score:.3f}<br/>"
                f"Value: {float(tx.get('value') or 0.0):.6f} ETH"
            ),
        }

    def get_transaction_chain(
        self,
        wallet: str,
        depth: int = 2,
        min_risk: float = 0.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit_nodes: int = 300,
        suspicious_only: bool = False,
    ) -> Dict[str, Any]:
        wallet = self._normalize_wallet(wallet)
        if not wallet:
            return {"nodes": [], "edges": [], "center": None, "alerts": []}

        depth = max(1, min(int(depth), 4))
        min_risk = self._clamp(min_risk)
        limit_nodes = max(50, min(int(limit_nodes), 1000))

        start_epoch = self._to_epoch_seconds(start_time)
        end_epoch = self._to_epoch_seconds(end_time)

        cache_key = (
            "chain",
            wallet,
            depth,
            round(min_risk, 4),
            start_epoch,
            end_epoch,
            int(suspicious_only),
            limit_nodes,
        )

        with self._lock:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

            nodes: Dict[str, Dict[str, Any]] = {}
            edges: List[Dict[str, Any]] = []
            edge_ids: Set[str] = set()
            seen_txs: Set[str] = set()

            queue: Deque[Tuple[str, int]] = deque([(wallet, 0)])
            visited_wallets: Set[str] = {wallet}

            while queue and len(nodes) < limit_nodes:
                current_wallet, hop = queue.popleft()
                wallet_id = f"wallet:{current_wallet}"
                if wallet_id not in nodes:
                    nodes[wallet_id] = self._wallet_node_locked(current_wallet)

                if hop >= depth:
                    continue

                tx_hashes = list(self._wallet_transactions.get(current_wallet, []))
                for tx_hash in tx_hashes:
                    tx = self._transactions.get(tx_hash)
                    if not tx:
                        continue
                    if not self._tx_passes_filters(tx, min_risk, start_time, end_time, suspicious_only):
                        continue

                    tx_id = f"tx:{tx_hash}"
                    if tx_id not in nodes:
                        nodes[tx_id] = self._tx_node_locked(tx)

                    if tx_hash not in seen_txs:
                        seen_txs.add(tx_hash)
                        for edge in self._tx_to_edges.get(tx_hash, []):
                            if edge["id"] in edge_ids:
                                continue
                            if edge["source"] not in nodes and edge["source"].startswith("wallet:"):
                                candidate_wallet = edge["source"].split("wallet:", 1)[1]
                                nodes[edge["source"]] = self._wallet_node_locked(candidate_wallet)
                            if edge["target"] not in nodes and edge["target"].startswith("wallet:"):
                                candidate_wallet = edge["target"].split("wallet:", 1)[1]
                                nodes[edge["target"]] = self._wallet_node_locked(candidate_wallet)
                            edge_ids.add(edge["id"])
                            edges.append(edge)

                    counterparty = self._counterparty_for_wallet(tx, current_wallet)
                    if counterparty and counterparty not in visited_wallets and hop + 1 <= depth:
                        visited_wallets.add(counterparty)
                        queue.append((counterparty, hop + 1))

                    if len(nodes) >= limit_nodes:
                        break

                if len(nodes) >= limit_nodes:
                    break

            node_ids = set(nodes.keys())
            filtered_edges = [
                edge for edge in edges if edge["source"] in node_ids and edge["target"] in node_ids
            ]

            result = {
                "center": f"wallet:{wallet}",
                "nodes": list(nodes.values()),
                "edges": filtered_edges,
                "alerts": self.get_wallet_alerts(wallet, limit=20),
            }

            self._cache_set(cache_key, result)
            return result

    def get_wallet_subgraph(
        self,
        wallet: str,
        depth: int = 2,
        min_risk: float = 0.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit_nodes: int = 300,
        suspicious_only: bool = False,
    ) -> Dict[str, Any]:
        wallet = self._normalize_wallet(wallet)
        if not wallet:
            return {
                "wallet": None,
                "nodes": [],
                "edges": [],
                "neighbors": [],
                "alerts": [],
                "metrics": {},
            }

        chain = self.get_transaction_chain(
            wallet=wallet,
            depth=depth,
            min_risk=min_risk,
            start_time=start_time,
            end_time=end_time,
            limit_nodes=limit_nodes,
            suspicious_only=suspicious_only,
        )

        metrics = {
            "base_risk_score": self.get_wallet_base_risk(wallet),
            "graph_risk_score": self.get_wallet_graph_risk(wallet),
            "temporal_score": self.get_wallet_temporal_score(wallet),
            "neighbor_count": self.get_neighbor_count(wallet),
            "tx_count": len(self._wallet_transactions.get(wallet, [])),
        }

        return {
            "wallet": wallet,
            "nodes": chain["nodes"],
            "edges": chain["edges"],
            "center": chain.get("center"),
            "neighbors": self.get_neighbors(wallet, limit=50),
            "alerts": chain.get("alerts", []),
            "metrics": metrics,
        }

    def find_shortest_path(
        self,
        source_wallet: str,
        target_wallet: str,
        max_depth: int = 8,
        directed: bool = True,
    ) -> Optional[List[str]]:
        source_wallet = self._normalize_wallet(source_wallet)
        target_wallet = self._normalize_wallet(target_wallet)
        if not source_wallet or not target_wallet:
            return None

        cache_key = ("path", source_wallet, target_wallet, int(max_depth), int(directed))
        with self._lock:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached.get("path")

            path = self._find_wallet_path_locked(
                source_wallet=source_wallet,
                target_wallet=target_wallet,
                max_depth=max_depth,
                directed=directed,
            )
            self._cache_set(cache_key, {"path": path})
            return path

    def get_path_edges(self, wallet_path: List[str]) -> List[Dict[str, Any]]:
        if not wallet_path or len(wallet_path) < 2:
            return []

        edges: List[Dict[str, Any]] = []
        now = datetime.utcnow()
        with self._lock:
            for index in range(len(wallet_path) - 1):
                source = self._normalize_wallet(wallet_path[index])
                target = self._normalize_wallet(wallet_path[index + 1])
                if not source or not target:
                    continue
                interaction = int(self._wallet_outgoing.get(source, {}).get(target, 0))
                if interaction == 0:
                    interaction = int(self._wallet_neighbors.get(source, {}).get(target, 0))
                edge = self._build_edge(
                    source=f"wallet:{source}",
                    target=f"wallet:{target}",
                    edge_type="TRANSFER",
                    weight=max(1.0, float(interaction)),
                    timestamp=now,
                    tx_hash=None,
                    metadata={"path_index": index, "interaction_count": interaction},
                )
                edges.append(edge)
        return edges
