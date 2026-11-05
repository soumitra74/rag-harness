import { useEffect, useState } from "react";
import {
  getSweepProgress,
  listSweeps,
  type SweepBudgetUsage,
  type SweepProgress,
  type SweepSummary,
} from "../api/client";

const POLL_INTERVAL_MS = 10_000;

function formatCap(cap: number | null): string {
  return cap == null ? "-" : String(cap);
}

function BudgetRow({ label, usage }: { label: string; usage: SweepBudgetUsage }) {
  return (
    <tr>
      <td>{label}</td>
      <td>
        ${usage.cost_used_usd.toFixed(2)} / ${formatCap(usage.cost_cap_usd)}
      </td>
      <td>
        {usage.minutes_used.toFixed(1)} / {formatCap(usage.minutes_cap)}
      </td>
      <td>
        {usage.iterations_used} / {formatCap(usage.iterations_cap)}
      </td>
    </tr>
  );
}

function SweepProgressView({ sweepId }: { sweepId: number }) {
  const [progress, setProgress] = useState<SweepProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = () =>
      getSweepProgress(sweepId)
        .then((result) => {
          if (cancelled) return;
          setProgress(result);
          setError(null);
        })
        .catch((err) => {
          if (!cancelled) setError((err as Error).message);
        });
    load();
    const timer = setInterval(load, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [sweepId]);

  if (error) return <p className="error">{error}</p>;
  if (!progress) return <p>Loading sweep status...</p>;

  const { current_config: current, latest_eval_cycle: cycle } = progress;

  return (
    <div className="sweep-status-panel">
      <p>Status: {progress.status}</p>
      <p>Configs evaluated: {progress.configs_evaluated}</p>
      <p>
        Current config:{" "}
        {current
          ? `#${current.config_id} (${current.status}) -- ${current.combos_done}/${current.combos_total} combos`
          : "none"}
      </p>
      <p>
        Latest eval cycle:{" "}
        {cycle
          ? `#${cycle.id} for config #${cycle.config_id} -- ${cycle.questions_written}/${cycle.questions_total} questions${cycle.completed ? " (completed)" : ""}`
          : "none"}
      </p>
      <table className="corpus-table">
        <thead>
          <tr>
            <th>Budget</th>
            <th>Cost (USD)</th>
            <th>Minutes</th>
            <th>Iterations</th>
          </tr>
        </thead>
        <tbody>
          <BudgetRow label="Ingestion" usage={progress.budgets.ingestion} />
          <BudgetRow label="Eval" usage={progress.budgets.eval} />
        </tbody>
      </table>
    </div>
  );
}

export function SweepStatusPanel() {
  const [sweeps, setSweeps] = useState<SweepSummary[] | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listSweeps()
      .then((list) => {
        setSweeps(list);
        setSelectedId((current) => current ?? list[0]?.sweep_id ?? null);
      })
      .catch((err) => setError((err as Error).message));
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (!sweeps) return <p>Loading sweeps...</p>;
  if (sweeps.length === 0) return <p>No sweep has been run yet.</p>;

  return (
    <div>
      <nav className="tabs">
        {sweeps.map((sweep) => (
          <button
            key={sweep.sweep_id}
            className={sweep.sweep_id === selectedId ? "tab active" : "tab"}
            onClick={() => setSelectedId(sweep.sweep_id)}
          >
            Sweep #{sweep.sweep_id}
          </button>
        ))}
      </nav>
      {selectedId !== null && <SweepProgressView key={selectedId} sweepId={selectedId} />}
    </div>
  );
}
