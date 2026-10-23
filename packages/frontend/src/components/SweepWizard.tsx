import { useState } from "react";
import { BootstrapReviewPanel } from "./wizard/BootstrapReviewPanel";
import { ResetDbPanel } from "./wizard/ResetDbPanel";
import { SuggestionPanel } from "./wizard/SuggestionPanel";
import { SweepProgressPanel } from "./wizard/SweepProgressPanel";
import { SweepResultPanel } from "./wizard/SweepResultPanel";
import { SweepSetupPanel } from "./wizard/SweepSetupPanel";

type WizardPage = 1 | 2 | 3 | 4 | 5 | 6;

interface SweepWizardProps {
  onExit: () => void;
  onExitToEvalHistory: () => void;
}

export function SweepWizard({ onExit, onExitToEvalHistory }: SweepWizardProps) {
  const [page, setPage] = useState<WizardPage>(1);
  const [sweepId, setSweepId] = useState<number | null>(null);

  return (
    <div className="sweep-wizard">
      <div className="sweep-wizard-header">
        <h2>Optimize RAG</h2>
        <button onClick={onExit}>Exit</button>
      </div>
      <div className="sweep-wizard-page">
        {page === 1 && (
          <div>
            <p>
              This wizard searches for an optimal RAG config for the current corpus, within a
              budget you set on the next page.
            </p>
            <ResetDbPanel />
            <div className="sweep-wizard-nav">
              <button onClick={() => setPage(2)}>Next</button>
            </div>
          </div>
        )}
        {page === 2 && (
          <SweepSetupPanel
            onStarted={(id) => {
              setSweepId(id);
              setPage(3);
            }}
          />
        )}
        {page === 3 && sweepId !== null && (
          <BootstrapReviewPanel sweepId={sweepId} onApproved={() => setPage(4)} />
        )}
        {page === 4 && sweepId !== null && (
          <SweepProgressPanel
            sweepId={sweepId}
            onAwaitingSuggestion={() => setPage(5)}
            onFinished={() => setPage(6)}
          />
        )}
        {page === 5 && sweepId !== null && (
          <SuggestionPanel sweepId={sweepId} onContinued={() => setPage(4)} />
        )}
        {page === 6 && sweepId !== null && (
          <SweepResultPanel sweepId={sweepId} onExitToEvalHistory={onExitToEvalHistory} />
        )}
      </div>
    </div>
  );
}
