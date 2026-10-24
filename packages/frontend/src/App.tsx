import { useEffect, useState } from "react";
import "./App.css";
import { getLatestSweep, type LatestSweep } from "./api/client";
import { ChunksPanel } from "./components/ChunksPanel";
import { CorpusLibrary } from "./components/CorpusLibrary";
import { EvalComparePanel } from "./components/EvalComparePanel";
import { EvalHistoryPanel } from "./components/EvalHistoryPanel";
import { EvalPanel } from "./components/EvalPanel";
import { GraphPanel } from "./components/GraphPanel";
import { IngestionHistoryPanel } from "./components/IngestionHistoryPanel";
import { IngestionPanel } from "./components/IngestionPanel";
import { RCAPanel } from "./components/RCAPanel";
import { RevalPanel } from "./components/RevalPanel";
import { SweepWizard } from "./components/SweepWizard";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8089";

const TABS = [
  { id: "corpus", label: "Corpus Library" },
  { id: "ingestion", label: "Ingestion" },
  { id: "history", label: "Ingestion History" },
  { id: "chunks", label: "Chunks" },
  { id: "graph", label: "Graph" },
  { id: "eval", label: "Eval" },
  { id: "reval", label: "Reval" },
  { id: "eval-history", label: "Eval History" },
  { id: "eval-compare", label: "Compare Evals" },
  { id: "rca", label: "RCA" },
] as const;

type TabId = (typeof TABS)[number]["id"];

const ACTIVE_TAB_STORAGE_KEY = "activeTab";

function isTabId(value: unknown): value is TabId {
  return TABS.some((tab) => tab.id === value);
}

function getStoredTab(): TabId {
  try {
    const stored = localStorage.getItem(ACTIVE_TAB_STORAGE_KEY);
    if (isTabId(stored)) return stored;
  } catch {
    // localStorage unavailable (private browsing, blocked site data, etc.) -- fall through.
  }
  return "corpus";
}

function App() {
  const [backendStatus, setBackendStatus] = useState("checking...");
  const [activeTab, setActiveTab] = useState<TabId>(getStoredTab);
  const [viewMode, setViewMode] = useState<"tabs" | "wizard">("tabs");
  const [latestSweep, setLatestSweep] = useState<LatestSweep | null>(null);
  const [resumeSweepId, setResumeSweepId] = useState<number | null>(null);

  useEffect(() => {
    if (viewMode !== "tabs") return;
    getLatestSweep().then(setLatestSweep).catch(() => setLatestSweep(null));
  }, [viewMode]);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/health`)
      .then((res) => res.json())
      .then((data) => setBackendStatus(data.status))
      .catch(() => setBackendStatus("unreachable"));
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(ACTIVE_TAB_STORAGE_KEY, activeTab);
    } catch {
      // Ignore -- persisting the tab is a convenience, not a requirement.
    }
  }, [activeTab]);

  return (
    <div className="app">
      <div className="app-header">
        <div>
          <h1>PDF RAG</h1>
          <p>Backend status: {backendStatus}</p>
        </div>
        {viewMode === "tabs" && (
          <div className="app-header-actions">
            {latestSweep?.resumable && (
              <button
                className="optimize-cta"
                onClick={() => {
                  setResumeSweepId(latestSweep.sweep_id);
                  setViewMode("wizard");
                }}
              >
                Resume Sweep
              </button>
            )}
            <button className="optimize-cta" onClick={() => setViewMode("wizard")}>
              Optimize RAG
            </button>
          </div>
        )}
      </div>
      {viewMode === "wizard" ? (
        <SweepWizard
          initialSweepId={resumeSweepId ?? undefined}
          onExit={() => {
            setResumeSweepId(null);
            setViewMode("tabs");
          }}
          onExitToEvalHistory={() => {
            setActiveTab("eval-history");
            setViewMode("tabs");
          }}
        />
      ) : (
        <>
          <nav className="tabs">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                className={tab.id === activeTab ? "tab active" : "tab"}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </nav>
          <div className="tab-content">
            {activeTab === "corpus" && <CorpusLibrary />}
            {activeTab === "ingestion" && <IngestionPanel />}
            {activeTab === "history" && <IngestionHistoryPanel />}
            {activeTab === "chunks" && <ChunksPanel />}
            {activeTab === "graph" && <GraphPanel />}
            {activeTab === "eval" && <EvalPanel />}
            {activeTab === "reval" && <RevalPanel />}
            {activeTab === "eval-history" && <EvalHistoryPanel />}
            {activeTab === "eval-compare" && <EvalComparePanel />}
            {activeTab === "rca" && <RCAPanel />}
          </div>
        </>
      )}
    </div>
  );
}

export default App;
