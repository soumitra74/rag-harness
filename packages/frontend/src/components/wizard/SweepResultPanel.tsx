import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { getSweep, type Sweep } from "../../api/client";

interface SweepResultPanelProps {
  sweepId: number;
  onExitToEvalHistory: () => void;
}

export function SweepResultPanel({ sweepId, onExitToEvalHistory }: SweepResultPanelProps) {
  const [sweep, setSweep] = useState<Sweep | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getSweep(sweepId)
      .then(setSweep)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [sweepId]);

  if (error) return <p className="error">{error}</p>;
  if (!sweep) return <p>Loading result...</p>;

  return (
    <div className="sweep-result-panel">
      <h3>Result</h3>
      <p>Status: {sweep.status}</p>
      {sweep.best_config_id !== null ? (
        <p>
          Best config: #{sweep.best_config_id}
          {sweep.best_score !== null && ` (score ${sweep.best_score.toFixed(3)})`}
        </p>
      ) : (
        <p>No config passed the quality gate.</p>
      )}

      {sweep.summary_markdown && (
        <div className="sweep-result-summary">
          <ReactMarkdown>{sweep.summary_markdown}</ReactMarkdown>
        </div>
      )}

      <button onClick={onExitToEvalHistory}>View in Eval History</button>
    </div>
  );
}
