import { useEffect, useState } from "react";
import { getSweep, suggestSweep, type Sweep } from "../../api/client";

interface SuggestionPanelProps {
  sweepId: number;
  onContinued: () => void;
}

const EMPTY_EXPLICIT_CONFIG = {
  chunking_strategy: "", chunk_size: "", chunk_overlap: "",
  embedding_provider: "", embedding_model: "", embedding_dim: "",
};

export function SuggestionPanel({ sweepId, onContinued }: SuggestionPanelProps) {
  const [sweep, setSweep] = useState<Sweep | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [hint, setHint] = useState("");
  const [explicitConfig, setExplicitConfig] = useState(EMPTY_EXPLICIT_CONFIG);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    getSweep(sweepId)
      .then(setSweep)
      .catch((err) => setLoadError(err instanceof Error ? err.message : String(err)));
  }, [sweepId]);

  function hasExplicitConfig(): boolean {
    return Object.values(explicitConfig).some((v) => v.trim() !== "");
  }

  async function handleAccept() {
    setSubmitting(true);
    setSubmitError(null);
    try {
      await suggestSweep(sweepId, { accept: true });
      onContinued();
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  }

  async function handleContinue() {
    setSubmitting(true);
    setSubmitError(null);
    try {
      await suggestSweep(sweepId, {
        hint: hint.trim() || undefined,
        explicitConfig: hasExplicitConfig()
          ? {
              chunking_strategy: explicitConfig.chunking_strategy,
              chunk_size: Number(explicitConfig.chunk_size),
              chunk_overlap: Number(explicitConfig.chunk_overlap),
              embedding_provider: explicitConfig.embedding_provider,
              embedding_model: explicitConfig.embedding_model,
              embedding_dim: Number(explicitConfig.embedding_dim),
            }
          : undefined,
      });
      onContinued();
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  }

  if (loadError) return <p className="error">{loadError}</p>;
  if (!sweep) return <p>Loading...</p>;

  return (
    <div className="suggestion-panel">
      <p>
        The search hasn't improved on the best score in a few tries. Current best:{" "}
        {sweep.best_score !== null ? sweep.best_score.toFixed(3) : "none yet"}
        {sweep.best_config_id !== null && ` (config #${sweep.best_config_id})`}
      </p>
      <h4>Attempts so far</h4>
      <ul>
        {sweep.sweep_configs.map((sc) => (
          <li key={sc.id}>
            Config #{sc.config_id} ({sc.source}, {sc.status}): {sc.rationale ?? "no rationale"}
          </li>
        ))}
      </ul>

      <label>
        Hint for the next candidate
        <textarea value={hint} onChange={(e) => setHint(e.target.value)} />
      </label>

      <fieldset>
        <legend>Or specify the next config directly</legend>
        {Object.entries(explicitConfig).map(([key, value]) => (
          <label key={key}>
            {key}
            <input
              type="text"
              value={value}
              onChange={(e) =>
                setExplicitConfig((prev) => ({ ...prev, [key]: e.target.value }))
              }
            />
          </label>
        ))}
      </fieldset>

      {submitError && <p className="error">{submitError}</p>}
      <div className="suggestion-actions">
        <button onClick={handleAccept} disabled={submitting}>
          Accept &amp; finalize
        </button>
        <button onClick={handleContinue} disabled={submitting}>
          Continue
        </button>
      </div>
    </div>
  );
}
