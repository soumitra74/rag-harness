import { useCallback, useEffect, useRef, useState } from "react";
import { approveSweep, getSweep, type Sweep } from "../../api/client";

const POLL_INTERVAL_MS = 2000;

interface BootstrapReviewPanelProps {
  sweepId: number;
  onApproved: () => void;
}

export function BootstrapReviewPanel({ sweepId, onApproved }: BootstrapReviewPanelProps) {
  const [sweep, setSweep] = useState<Sweep | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [approving, setApproving] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const result = await getSweep(sweepId);
        if (cancelled) return;
        setSweep(result);
        if (result.status !== "bootstrapping") stopPolling();
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          stopPolling();
        }
      }
    }

    poll();
    pollRef.current = setInterval(poll, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
    };
  }, [sweepId, stopPolling]);

  async function handleApprove() {
    setApproving(true);
    setError(null);
    try {
      await approveSweep(sweepId);
      onApproved();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setApproving(false);
    }
  }

  if (error) return <p className="error">{error}</p>;
  if (!sweep || sweep.status === "bootstrapping") {
    return <p>Inspecting corpus and preparing candidate configs...</p>;
  }
  if (sweep.status === "failed") {
    return <p className="error">Bootstrap failed: {sweep.error_message}</p>;
  }

  return (
    <div className="bootstrap-review-panel">
      <h3>Candidate configs</h3>
      <ul className="bootstrap-candidates">
        {sweep.sweep_configs.map((sc) => (
          <li key={sc.id}>
            Config #{sc.config_id} ({sc.source}): {sc.rationale ?? "no rationale given"}
          </li>
        ))}
      </ul>
      <button onClick={handleApprove} disabled={approving}>
        {approving ? "Approving..." : "Approve"}
      </button>
    </div>
  );
}
