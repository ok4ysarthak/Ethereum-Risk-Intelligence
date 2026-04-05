import React, { useEffect, useMemo, useRef, useState } from "react";
import { DataSet, Network } from "vis-network/standalone";

const riskColor = (score) => {
  const value = Number(score || 0);
  if (value >= 0.7) return "#ef4444";
  if (value >= 0.4) return "#f59e0b";
  return "#22c55e";
};

export default function GraphView({ baseUrl = "" }) {
  const containerRef = useRef(null);
  const networkRef = useRef(null);
  const nodesRef = useRef(new DataSet());
  const edgesRef = useRef(new DataSet());

  const [wallet, setWallet] = useState("");
  const [depth, setDepth] = useState(2);
  const [minRisk, setMinRisk] = useState(0);
  const [status, setStatus] = useState("Ready");
  const [selection, setSelection] = useState(null);

  const normalizedWallet = useMemo(() => {
    const value = String(wallet || "").trim().toLowerCase();
    if (!value) return "";
    return value.startsWith("0x") ? value : `0x${value}`;
  }, [wallet]);

  useEffect(() => {
    if (!containerRef.current || networkRef.current) return;

    networkRef.current = new Network(
      containerRef.current,
      {
        nodes: nodesRef.current,
        edges: edgesRef.current,
      },
      {
        interaction: {
          hover: true,
          navigationButtons: true,
        },
        physics: {
          enabled: true,
          barnesHut: {
            gravitationalConstant: -4500,
            springLength: 140,
            damping: 0.2,
          },
        },
      }
    );

    networkRef.current.on("click", (params) => {
      if (params.nodes.length) {
        const node = nodesRef.current.get(params.nodes[0]);
        setSelection({ type: "node", payload: node?.data || node });
        return;
      }
      if (params.edges.length) {
        const edge = edgesRef.current.get(params.edges[0]);
        setSelection({ type: "edge", payload: edge?.data || edge });
        return;
      }
      setSelection(null);
    });
  }, []);

  const loadGraph = async () => {
    if (!normalizedWallet) {
      setStatus("Enter a wallet address first.");
      return;
    }

    setStatus("Loading graph...");
    try {
      const params = new URLSearchParams({
        depth: String(depth),
        min_risk: String(minRisk),
      });

      const response = await fetch(
        `${baseUrl}/graph/wallet/${normalizedWallet}?${params.toString()}`
      );
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = await response.json();

      const visNodes = (payload.nodes || []).map((node) => {
        if (node.type === "transaction") {
          return {
            id: node.id,
            label: node.label || node.tx_hash,
            shape: "box",
            color: { background: riskColor(node.risk_score), border: "#111827" },
            font: { color: "#f5f5f5" },
            data: node,
          };
        }
        return {
          id: node.id,
          label: node.label || node.address,
          shape: "dot",
          size: node.size || 20,
          color: { background: riskColor(node.graph_risk_score), border: "#111827" },
          font: { color: "#e5e7eb" },
          data: node,
        };
      });

      const visEdges = (payload.edges || []).map((edge) => ({
        id: edge.id,
        from: edge.source,
        to: edge.target,
        arrows: "to",
        width: Math.min(4, 1 + Number(edge.weight || 1)),
        color: {
          color:
            edge.edge_type === "FUNDING"
              ? "#fbbf24"
              : edge.edge_type === "RELATED"
              ? "#60a5fa"
              : "#a1a1aa",
          opacity: 0.9,
        },
        dashes: edge.edge_type === "RELATED",
        label: edge.edge_type,
        font: { color: "#9ca3af", size: 10 },
        data: edge,
      }));

      nodesRef.current.clear();
      edgesRef.current.clear();
      nodesRef.current.add(visNodes);
      edgesRef.current.add(visEdges);

      if (payload.center) {
        networkRef.current.focus(payload.center, {
          scale: 1,
          animation: true,
        });
      } else {
        networkRef.current.fit({ animation: true });
      }

      setStatus(`Loaded ${visNodes.length} nodes and ${visEdges.length} edges.`);
    } catch (error) {
      setStatus(`Failed to load graph: ${error.message}`);
    }
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "3fr 1fr", gap: 16 }}>
      <div>
        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr auto", gap: 8, marginBottom: 8 }}>
          <input
            value={wallet}
            onChange={(event) => setWallet(event.target.value)}
            placeholder="Wallet address"
          />
          <input
            type="number"
            min={1}
            max={4}
            value={depth}
            onChange={(event) => setDepth(Number(event.target.value || 2))}
          />
          <input
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={minRisk}
            onChange={(event) => setMinRisk(Number(event.target.value || 0))}
          />
          <button onClick={loadGraph}>Load</button>
        </div>
        <div ref={containerRef} style={{ width: "100%", height: "70vh", border: "1px solid #333" }} />
        <p style={{ marginTop: 8 }}>{status}</p>
      </div>

      <div>
        <h3>Selection</h3>
        <pre style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>
          {selection ? JSON.stringify(selection.payload, null, 2) : "Select a node or edge"}
        </pre>
      </div>
    </div>
  );
}
