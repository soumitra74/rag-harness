# RAG Harness Frontend

React and TypeScript UI for the RAG harness: the sweep wizard, sweep progress and results, eval runs, and chunk inspection.

## Development

```
npm install
npm run dev
```

The dev server runs on http://localhost:5183. It calls the backend at `http://localhost:8089` by default; set `VITE_API_BASE_URL` to point elsewhere.

## Scripts

| Command           | Purpose                          |
|-------------------|----------------------------------|
| `npm run dev`     | Start the Vite dev server        |
| `npm run build`   | Type-check and build for production |
| `npm run preview` | Serve the production build       |
| `npm run lint`    | Run oxlint                       |
