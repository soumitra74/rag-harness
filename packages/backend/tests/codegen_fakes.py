class CodegenFakeMethods:
    @property
    def codegen_runs(self):
        return self.__dict__.setdefault("_codegen_runs", [])

    @property
    def generated_artifacts(self):
        return self.__dict__.setdefault("_generated_artifacts", {})

    async def sum_codegen_spend(self, sweep_id):
        return {
            "cost_usd": sum(r["cost_usd"] for r in self.codegen_runs),
            "minutes": sum(r["duration_ms"] for r in self.codegen_runs) / 60000.0,
            "attempts": len(self.codegen_runs),
        }

    async def insert_codegen_run(self, sweep_id, task, attempt, passed, error, cost_usd, duration_ms):
        self.codegen_runs.append({
            "sweep_id": sweep_id, "task": task, "attempt": attempt, "passed": passed,
            "error": error, "cost_usd": cost_usd, "duration_ms": duration_ms,
        })

    async def upsert_generated_artifact(self, sweep_id, task, content):
        self.generated_artifacts[task] = content

    async def get_generated_artifacts(self, sweep_id):
        return dict(self.generated_artifacts)
