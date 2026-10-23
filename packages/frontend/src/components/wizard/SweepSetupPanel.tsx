import { useState } from "react";
import {
  startSweep,
  type CodegenRequest,
  type CodegenTaskName,
  type SweepBudget,
} from "../../api/client";

const DEFAULT_INGESTION_BUDGET: SweepBudget = { cost_usd: 10, minutes: 60, iterations: 10 };
const DEFAULT_EVAL_BUDGET: SweepBudget = { cost_usd: 10, minutes: 30, iterations: 50 };
const DEFAULT_CODEGEN_BUDGET: CodegenRequest = { cost_usd: 2, minutes: 15, attempts: 6, tasks: [] };
const CODEGEN_TASK_LABELS: Record<CodegenTaskName, string> = {
  ocr: "OCR for scanned pages (Tesseract)",
  graph_labels: "Graph entity labels for this corpus",
};

interface SweepSetupPanelProps {
  onStarted: (sweepId: number) => void;
}

function BudgetFields({
  label,
  value,
  onChange,
}: {
  label: string;
  value: SweepBudget;
  onChange: (next: SweepBudget) => void;
}) {
  return (
    <fieldset className="sweep-budget-fields">
      <legend>{label}</legend>
      <label>
        Cost ($)
        <input
          type="number"
          min="0"
          value={value.cost_usd}
          onChange={(e) => onChange({ ...value, cost_usd: Number(e.target.value) })}
        />
      </label>
      <label>
        Time (minutes)
        <input
          type="number"
          min="0"
          value={value.minutes}
          onChange={(e) => onChange({ ...value, minutes: Number(e.target.value) })}
        />
      </label>
      <label>
        Iterations
        <input
          type="number"
          min="0"
          value={value.iterations}
          onChange={(e) => onChange({ ...value, iterations: Number(e.target.value) })}
        />
      </label>
    </fieldset>
  );
}

export function SweepSetupPanel({ onStarted }: SweepSetupPanelProps) {
  const [ingestionBudget, setIngestionBudget] = useState(DEFAULT_INGESTION_BUDGET);
  const [evalBudget, setEvalBudget] = useState(DEFAULT_EVAL_BUDGET);
  const [generateGoldenQa, setGenerateGoldenQa] = useState(true);
  const [codegenEnabled, setCodegenEnabled] = useState(false);
  const [codegen, setCodegen] = useState(DEFAULT_CODEGEN_BUDGET);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleStart() {
    setStarting(true);
    setError(null);
    try {
      const result = await startSweep(
        ingestionBudget,
        evalBudget,
        generateGoldenQa,
        codegenEnabled ? codegen : null,
      );
      onStarted(result.sweep_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStarting(false);
    }
  }

  return (
    <div className="sweep-setup-panel">
      <label className="golden-qa-toggle">
        <input
          type="checkbox"
          checked={generateGoldenQa}
          onChange={(e) => setGenerateGoldenQa(e.target.checked)}
        />
        Generate a golden QA set if one doesn't already exist
      </label>

      <BudgetFields label="Ingestion budget" value={ingestionBudget} onChange={setIngestionBudget} />
      <BudgetFields label="Eval budget" value={evalBudget} onChange={setEvalBudget} />

      <label className="codegen-toggle">
        <input
          type="checkbox"
          checked={codegenEnabled}
          onChange={(e) => setCodegenEnabled(e.target.checked)}
        />
        Generate code before the sweep (optional)
      </label>

      {codegenEnabled && (
        <fieldset className="sweep-budget-fields codegen-fields">
          <legend>Code generation</legend>
          {(Object.keys(CODEGEN_TASK_LABELS) as CodegenTaskName[]).map((task) => (
            <label key={task}>
              <input
                type="checkbox"
                checked={codegen.tasks.includes(task)}
                onChange={(e) =>
                  setCodegen({
                    ...codegen,
                    tasks: e.target.checked
                      ? [...codegen.tasks, task]
                      : codegen.tasks.filter((t) => t !== task),
                  })
                }
              />
              {CODEGEN_TASK_LABELS[task]}
            </label>
          ))}
          <label>
            Cost ($)
            <input
              type="number"
              min="0"
              value={codegen.cost_usd}
              onChange={(e) => setCodegen({ ...codegen, cost_usd: Number(e.target.value) })}
            />
          </label>
          <label>
            Time (minutes)
            <input
              type="number"
              min="0"
              value={codegen.minutes}
              onChange={(e) => setCodegen({ ...codegen, minutes: Number(e.target.value) })}
            />
          </label>
          <label>
            Attempts
            <input
              type="number"
              min="1"
              value={codegen.attempts}
              onChange={(e) => setCodegen({ ...codegen, attempts: Number(e.target.value) })}
            />
          </label>
          <p className="hint">If code generation fails, the sweep stops before any configs are ingested.</p>
        </fieldset>
      )}

      {error && <p className="error">{error}</p>}
      <button onClick={handleStart} disabled={starting || (codegenEnabled && codegen.tasks.length === 0)}>
        {starting ? "Starting..." : "Start sweep"}
      </button>
    </div>
  );
}
