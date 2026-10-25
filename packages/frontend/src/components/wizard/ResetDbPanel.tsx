import { useState } from "react";
import { executeReset, previewReset } from "../../api/client";

interface Category {
  id: string;
  label: string;
  requires: string[];
}

const CATEGORIES: Category[] = [
  { id: "ingestion_data", label: "Ingestion data (configs + chunks + ingestion runs)", requires: ["graph_data", "eval_history"] },
  { id: "documents", label: "Documents", requires: ["graph_data", "golden_qa"] },
  { id: "graph_data", label: "Graph data", requires: [] },
  { id: "golden_qa", label: "Golden QA dataset", requires: ["eval_history"] },
  { id: "eval_history", label: "Eval history", requires: ["rca_sessions"] },
  { id: "rca_sessions", label: "RCA sessions", requires: [] },
  { id: "generated_code", label: "Generated code (DB records + generated/ files)", requires: [] },
];

const REQUIRED_PHRASE = "RESET";

function resolveCategories(selected: Set<string>): Set<string> {
  const resolved = new Set<string>();
  const stack = [...selected];
  while (stack.length > 0) {
    const id = stack.pop()!;
    if (resolved.has(id)) continue;
    resolved.add(id);
    const category = CATEGORIES.find((c) => c.id === id);
    if (category) stack.push(...category.requires);
  }
  return resolved;
}

export function ResetDbPanel() {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [preview, setPreview] = useState<Record<string, number> | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [confirmPhrase, setConfirmPhrase] = useState("");
  const [executeResult, setExecuteResult] = useState<string | null>(null);
  const [executeError, setExecuteError] = useState<string | null>(null);

  const resolved = resolveCategories(selected);

  function toggle(id: string) {
    const next = new Set(selected);
    if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
    }
    setSelected(next);
    setPreview(null);
    setExecuteResult(null);
  }

  async function handlePreview() {
    setPreviewError(null);
    try {
      setPreview(await previewReset(Array.from(resolved)));
    } catch (err) {
      setPreviewError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleExecute() {
    setExecuteError(null);
    try {
      const result = await executeReset(Array.from(resolved), confirmPhrase);
      setExecuteResult(`Reset complete. Backup written to ${result.backup_path}`);
      setSelected(new Set());
      setPreview(null);
      setConfirmPhrase("");
    } catch (err) {
      setExecuteError(err instanceof Error ? err.message : String(err));
    }
  }

  const canExecute = resolved.size > 0 && confirmPhrase === REQUIRED_PHRASE;

  return (
    <div className="reset-db-panel">
      <h3>Reset DB</h3>
      <p>
        Starting a new project? Wipe existing RAG data so the sweep starts from a clean slate.
        This is destructive -- a backup is written before anything is deleted.
      </p>
      <ul className="reset-db-categories">
        {CATEGORIES.map((category) => {
          const checked = resolved.has(category.id);
          const disabled = checked && !selected.has(category.id);
          return (
            <li key={category.id}>
              <label>
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={disabled}
                  onChange={() => toggle(category.id)}
                />
                {category.label}
                {disabled && <span className="reset-db-required-note"> (required by another selection)</span>}
              </label>
            </li>
          );
        })}
      </ul>

      <button onClick={handlePreview} disabled={resolved.size === 0}>
        Preview
      </button>
      {previewError && <p className="error">{previewError}</p>}
      {preview && (
        <ul className="reset-db-preview">
          {Object.entries(preview).map(([table, count]) => (
            <li key={table}>
              {table}: {count} row(s)
            </li>
          ))}
        </ul>
      )}

      <div className="reset-db-confirm">
        <label>
          Type "{REQUIRED_PHRASE}" to confirm:
          <input
            type="text"
            value={confirmPhrase}
            onChange={(e) => setConfirmPhrase(e.target.value)}
          />
        </label>
        <button onClick={handleExecute} disabled={!canExecute}>
          Confirm reset
        </button>
      </div>
      {executeError && <p className="error">{executeError}</p>}
      {executeResult && <p>{executeResult}</p>}
    </div>
  );
}
