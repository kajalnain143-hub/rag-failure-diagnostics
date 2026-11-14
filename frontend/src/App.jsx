import { useEffect, useState } from "react";

const API = "http://localhost:8000";

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [expected, setExpected] = useState("");
  const [result, setResult] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [incident, setIncident] = useState("");
  const [incidentLogs, setIncidentLogs] = useState("");
  const [expectedBehavior, setExpectedBehavior] = useState("");
  const [incidentResult, setIncidentResult] = useState(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  async function loadDocuments() {
    try {
      const response = await fetch(`${API}/api/documents`);
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch {
      setMessage("Cannot reach the API. Start the FastAPI backend first.");
    }
  }

  useEffect(() => { loadDocuments(); }, []);

  async function uploadDocument(event) {
    event.preventDefault();
    if (!file) return setMessage("Choose a PDF, TXT, or Markdown file first.");
    const form = new FormData();
    form.append("file", file);
    setBusy(true);
    setMessage("");
    try {
      const response = await fetch(`${API}/api/documents/upload`, { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Upload failed.");
      setMessage(`Uploaded ${data.filename} — indexed ${data.chunk_count} chunks.`);
      setFile(null);
      event.target.reset();
      await loadDocuments();
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function runQuery(event) {
    event.preventDefault();
    if (!question.trim()) return;
    setBusy(true);
    setMessage("");
    setEvaluation(null);
    try {
      const response = await fetch(`${API}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 4 })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Query failed.");
      setResult(data);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function runEvaluation() {
    if (!question.trim() || !expected.trim()) {
      setMessage("Enter a question and an expected answer to evaluate retrieval.");
      return;
    }
    setBusy(true);
    try {
      const response = await fetch(`${API}/api/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, expected_answer: expected, top_k: 4 })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Evaluation failed.");
      setEvaluation(data);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function runIncidentDiagnosis(event) {
    event.preventDefault();
    if (incident.trim().length < 10) {
      setMessage("Describe the incident in at least 10 characters.");
      return;
    }
    setBusy(true);
    setMessage("");
    setIncidentResult(null);
    try {
      const response = await fetch(`${API}/api/diagnostics/incident`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          description: incident,
          logs: incidentLogs,
          expected_behavior: expectedBehavior
        })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Incident diagnosis failed.");
      setIncidentResult(data.diagnosis);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <header className="hero">
        <div className="eyebrow">AI RELIABILITY WORKBENCH</div>
        <h1>RAG Failure <span>Diagnostics</span></h1>
        <p>Inspect retrieval quality, diagnose incidents, and evaluate your RAG pipeline.</p>
        <div className="status"><i /> Gemini-grounded answers · Incident triage · Retrieval baseline</div>
      </header>

      {message && <div className="notice">{message}</div>}

      <section className="panel">
        <div className="section-heading"><div><div className="eyebrow">01 / KNOWLEDGE BASE</div><h2>Upload documents</h2></div><span className="count">{documents.length} indexed</span></div>
        <form className="upload-row" onSubmit={uploadDocument}>
          <input type="file" accept=".pdf,.txt,.md" onChange={e => setFile(e.target.files[0])} />
          <button disabled={busy}>{busy ? "Working..." : "Upload & index"}</button>
        </form>
        <p className="hint">Supported formats: PDF, TXT, Markdown. Documents are stored in the backend SQLite database.</p>
        {documents.length > 0 && <div className="doc-list">{documents.map(doc => <div className="doc" key={doc.id}><span>▤ {doc.filename}</span><small>{doc.chunk_count} chunks</small></div>)}</div>}
      </section>

      <section className="panel">
        <div className="eyebrow">02 / RETRIEVAL INSPECTION</div>
        <h2>Ask your knowledge base</h2>
        <form onSubmit={runQuery}>
          <textarea value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask a question about your uploaded documents..." rows="3" />
          <button disabled={busy}>{busy ? "Analyzing..." : "Run retrieval diagnostics"}</button>
        </form>
        {result && <div className="results">
          <div className="answer"><div className="eyebrow">ANSWER · {result.answer_mode?.replaceAll("_", " ")}</div><p>{result.answer}</p></div>
          <div className="diagnostic"><div><div className="eyebrow">DIAGNOSTIC STATUS</div><strong>{result.diagnostics.status.replaceAll("_", " ")}</strong></div><div className="score">{result.diagnostics.best_similarity_score ?? 0}<small>best similarity</small></div></div>
          {result.sources?.length > 0 && <><h3>Answer sources</h3>{result.sources.map(source => <article className="chunk" key={source.label}><div><b>[{source.label}] {source.filename}</b><span>Score: {source.score}</span></div><p>{source.content}</p></article>)}</>}
          {result.diagnostics.issues?.map((item, i) => <p className="issue" key={i}>⚠ {item}</p>)}
          {result.diagnostics.recommendations?.map((item, i) => <p className="recommendation" key={i}>→ {item}</p>)}
          <h3>Retrieved chunks</h3>
          {result.retrieved_chunks.map(chunk => <article className="chunk" key={chunk.chunk_id}><div><b>{chunk.filename}</b><span>Score: {chunk.score}</span></div><p>{chunk.content}</p></article>)}
          <p className="hint">{result.diagnostics.note}</p>
        </div>}
      </section>

      <section className="panel">
        <div className="eyebrow">03 / INCIDENT TRIAGE CLINIC</div>
        <h2>Diagnose a RAG / LLM failure</h2>
        <p className="hint">Describe an incident. Gemini maps it to one primary failure pattern (P01–P12), identifies missing evidence, and suggests a minimal structural fix.</p>
        <form onSubmit={runIncidentDiagnosis}>
          <textarea value={incident} onChange={e => setIncident(e.target.value)} placeholder="Example: The retrieved chunks clearly say refunds are not allowed, but the assistant confidently tells users they can get a refund..." rows="4" />
          <textarea value={expectedBehavior} onChange={e => setExpectedBehavior(e.target.value)} placeholder="Expected behavior (optional)" rows="2" />
          <textarea value={incidentLogs} onChange={e => setIncidentLogs(e.target.value)} placeholder="Relevant logs / traces (optional)" rows="3" />
          <button disabled={busy}>{busy ? "Diagnosing..." : "Run incident diagnosis"}</button>
        </form>
        {incidentResult && <div className="results">
          <div className="diagnostic">
            <div><div className="eyebrow">PRIMARY FAILURE PATTERN</div><strong>{incidentResult.primary_pattern_id} · {incidentResult.severity?.toUpperCase()}</strong></div>
            <div className="score">{Math.round(incidentResult.confidence * 100)}%<small>model confidence</small></div>
          </div>
          <h3>{incidentResult.primary_pattern_id} diagnosis</h3>
          <p>{incidentResult.reasoning?.join(" ")}</p>
          {incidentResult.secondary_pattern_ids?.length > 0 && <p className="hint">Secondary candidates: {incidentResult.secondary_pattern_ids.join(", ")}</p>}
          <h3>Minimal structural fix</h3>
          {incidentResult.minimal_structural_fix?.map((item, i) => <p className="recommendation" key={i}>→ {item}</p>)}
          <h3>Missing evidence</h3>
          {incidentResult.missing_evidence?.map((item, i) => <p className="issue" key={i}>? {item}</p>)}
          <h3>Verification steps</h3>
          {incidentResult.verification_steps?.map((item, i) => <p className="recommendation" key={i}>✓ {item}</p>)}
          <p className="hint">{incidentResult.disclaimer}</p>
        </div>}
      </section>

      <section className="panel">
        <div className="eyebrow">04 / BASELINE EVALUATION</div>
        <h2>Evaluate retrieval coverage</h2>
        <p className="hint">Provide a reference answer. This baseline checks whether its key terms appear in retrieved context.</p>
        <textarea value={expected} onChange={e => setExpected(e.target.value)} placeholder="Enter the expected answer..." rows="3" />
        <button className="secondary" onClick={runEvaluation} disabled={busy}>Evaluate retrieval</button>
        {evaluation && <div className="evaluation"><strong>{(evaluation.evaluation.expected_answer_term_coverage * 100).toFixed(1)}%</strong><span>Expected-answer term coverage</span><p>{evaluation.evaluation.interpretation}</p></div>}
      </section>
      <footer>RAG Failure Diagnostics · Gemini grounded generation · Incident clinic</footer>
    </main>
  );
}
