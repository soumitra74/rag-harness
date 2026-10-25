import { useCallback, useEffect, useRef, useState } from "react";
import { getSweep, resumeSweep, type Sweep } from "../../api/client";

const POLL_INTERVAL_MS = 2000;
const TERMINAL_STATUSES = new Set(["stopped_budget", "stopped_optimal"]);

interface SweepProgressPanelProps {
  sweepId: number;
  onAwaitingApproval: () => void;
  onAwaitingSuggestion: () => void;
  onFinished: () => void;
}

function formatBudget(spent: { cost_usd: number; minutes: number; iterations: number }, budget: {
  cost_usd: number; minutes: number; iterations: number;
}): string {
  return (
    `$${spent.cost_usd.toFixed(2)}/$${budget.cost_usd} -- ` +
    `${spent.minutes.toFixed(1)}/${budget.minutes} min -- ` +
    `${spent.iterations}/${budget.iterations} iterations`
  );
}

export function SweepProgressPanel({
  sweepId, onAwaitingApproval, onAwaitingSuggestion, onFinished,
}: SweepProgressPanelProps) {
  const [sweep, setSweep] = useState<Sweep | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [resuming, setResuming] = useState(false);
  const [resumeError, setResumeError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  // One poll function reused by the initial mount and by handleResume, so
  // resuming re-enters the same status-transition logic instead of its own
  // narrower "just watch for paused to end" loop that never continued on
  // to awaiting_suggestion/finished (review finding I-9).
  const poll = useCallback(async (): Promise<boolean> => {
    try {
      const result = await getSweep(sweepId);
      setSweep(result);
      if (result.status === "awaiting_approval") {
        stopPolling();
        onAwaitingApproval();
        return true;
      }
      if (result.status === "awaiting_suggestion") {
        stopPolling();
        onAwaitingSuggestion();
        return true;
      }
      if (TERMINAL_STATUSES.has(result.status)) {
        stopPolling();
        onFinished();
        return true;
      }
      if (result.status === "paused" || result.status === "failed") {
        stopPolling();
        return true;
      }
      return false;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      stopPolling();
      return true;
    }
    // onAwaitingSuggestion/onFinished intentionally excluded: SweepWizard
    // passes fresh inline closures each render, and including them would
    // restart polling on every parent re-render. Both callbacks only call
    // setPage with a literal, so a "stale" reference is behaviorally
    // identical to a fresh one.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sweepId, stopPolling]);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      const done = await poll();
      if (done || cancelled) return;
      pollRef.current = setInterval(poll, POLL_INTERVAL_MS);
    })();

    return () => {
      cancelled = true;
    };
  }, [poll]);

  async function handleResume() {
    setResuming(true);
    setResumeError(null);
    try {
      await resumeSweep(sweepId);
      setResuming(false);
      const done = await poll();
      if (!done) {
        pollRef.current = setInterval(poll, POLL_INTERVAL_MS);
      }
    } catch (err) {
      setResumeError(err instanceof Error ? err.message : String(err));
      setResuming(false);
    }
  }

  if (error) return <p className="error">{error}</p>;
  if (!sweep) return <p>Loading sweep status...</p>;

  return (
    <div className="sweep-progress-panel">
      <p>Status: {sweep.status}</p>
      {sweep.status === "failed" && <p className="error">Sweep failed: {sweep.error_message}</p>}
      <p>Ingestion budget: {formatBudget(sweep.ingestion_spent, sweep.ingestion_budget)}</p>
      <p>Eval budget: {formatBudget(sweep.eval_spent, sweep.eval_budget)}</p>
      {sweep.best_score !== null && <p>Best score so far: {sweep.best_score.toFixed(3)}</p>}

      {(sweep.status === "paused" || sweep.status === "failed") && (
        <div>
          <button onClick={handleResume} disabled={resuming}>
            {resuming ? "Resuming..." : sweep.status === "failed" ? "Retry" : "Resume"}
          </button>
          {resumeError && <p className="error">{resumeError}</p>}
        </div>
      )}
    </div>
  );
}
