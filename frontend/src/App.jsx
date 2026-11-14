import { useEffect, useState } from "react";

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

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
  const [incidentHistory, setIncidentHistory] = useState([]);
  const [incidentSummary, setIncidentSummary] = useState({ total_incidents: 0, patterns: [], severity: {} });
  const [incidentPatternFilter, setIncidentPatternFilter] = useState("");
  const [incidentSeverityFilter, setIncidentSeverityFilter] = useState("");
  const [experiments, setExperiments] = useState([]);
  const [benchmarkCase, setBenchmarkCase] = useState(JSON.stringify({
    query_id: "my-query-001",
    question: "What does the document say about the main topic?",
    relevant_chunk_ids: [1],
    reference_answer: "Write the expected answer here.",
    k: 5
  }, null, 2));
  const [benchmarkResult, setBenchmarkResult] = useState(null);
  const [traces, setTraces] = useState([]);
  const [selectedTrace, setSelectedTrace] = useState(null);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [batchDataset, setBatchDataset] = useState(JSON.stringify({
    dataset_name: "my-rag-regression-suite",
    strategies: ["tfidf", "hybrid", "hybrid_mmr"],
    mmr_diversity: 0.35,
    cases: [{ query_id: "q1", question: "Ask a question about your indexed documents", relevant_chunk_ids: [1], reference_answer: "Replace with the expected answer", k: 5 }]
  }, null, 2));
  const [batchResult, setBatchResult] = useState(null);
  const [baselineRunId, setBaselineRunId] = useState("");
  const [currentRunId, setCurrentRunId] = useState("");
  const [regressionResult, setRegressionResult] = useState(null);
  const [reportDatasetName, setReportDatasetName] = useState("my-rag-regression-suite");
  const [reportStrategy, setReportStrategy] = useState("hybrid");
  const [benchmarkReport, setBenchmarkReport] = useState(null);
  const [benchmarkInsights, setBenchmarkInsights] = useState(null);
  const [failureAnalytics, setFailureAnalytics] = useState(null);
  const [replayBusy, setReplayBusy] = useState(false);
  const [qualityResult, setQualityResult] = useState(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [dashboardBusy, setDashboardBusy] = useState(false);

  async function apiRequest(path, options = {}) {
    const response = await fetch(`${API}${path}`, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || data.message || `Request failed (${response.status})`);
    return data;
  }

  async function loadDocuments() {
    try {
      const data = await apiRequest("/api/documents");
      setDocuments(data.documents || []);
    } catch {
      setMessage("Cannot reach the API. Start the FastAPI backend first.");
    }
  }

  async function loadExperiments() {
    const data = await apiRequest("/api/experiments?limit=30");
    setExperiments(data.experiments || []);
  }

  async function loadTraces() {
    const data = await apiRequest("/api/traces?limit=30");
    setTraces(data.traces || []);
  }

  async function inspectTrace(traceId) {
    try {
      const data = await apiRequest(`/api/traces/${traceId}`);
      setSelectedTrace(data.trace);
    } catch (error) {
      setMessage(error.message);
    }
  }

  useEffect(() => {
    loadDocuments();
    loadExperiments().catch(() => {});
    loadTraces().catch(() => {});
    loadIncidentHistory().catch(() => {});
  }, []);

  async function uploadDocument(event) {
    event.preventDefault();
    if (!file) return setMessage("Choose a PDF, TXT, or Markdown file first.");
    const form = new FormData();
    form.append("file", file);
    setBusy(true);
    setMessage("");
    try {
      const data = await apiRequest("/api/documents/upload", { method: "POST", body: form });
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
      const data = await apiRequest("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 4 })
      });
      setResult(data);
      await loadTraces();
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
    setMessage("");
    try {
      const data = await apiRequest("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, expected_answer: expected, top_k: 4 })
      });
      setEvaluation(data);
      await apiRequest("/api/experiments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: `Manual evaluation · ${question.slice(0, 54)}`,
          config: { method: "hybrid-retrieval", top_k: 4, question },
          metrics: data.evaluation,
          notes: "Saved automatically from the baseline evaluation panel."
        })
      });
      await loadExperiments();
      setMessage("Evaluation complete and experiment saved to history.");
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function runBenchmark(event) {
    event.preventDefault();
    let testCase;
    try {
      testCase = JSON.parse(benchmarkCase);
    } catch {
      setMessage("Benchmark case must be valid JSON.");
      return;
    }
    setDashboardBusy(true);
    setMessage("");
    setBenchmarkResult(null);
    try {
      const data = await apiRequest("/api/benchmark/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(testCase)
      });
      setBenchmarkResult(data);
      await apiRequest("/api/experiments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: `Benchmark · ${testCase.query_id}`,
          config: { method: "current-retriever", k: testCase.k || 5, query_id: testCase.query_id },
          metrics: data.metrics,
          notes: "Saved automatically from the labeled benchmark panel."
        })
      });
      await loadExperiments();
      setMessage("Benchmark completed and saved to experiment history.");
    } catch (error) {
      setMessage(error.message);
    } finally {
      setDashboardBusy(false);
    }
  }

  async function compareStrategies() {
    let testCase;
    try {
      testCase = JSON.parse(benchmarkCase);
    } catch {
      setMessage("Benchmark case must be valid JSON.");
      return;
    }
    setDashboardBusy(true);
    setMessage("");
    setComparisonResult(null);
    try {
      const data = await apiRequest("/api/benchmark/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(testCase)
      });
      setComparisonResult(data);
      await Promise.all(Object.entries(data.comparison).map(([strategy, result]) =>
        apiRequest("/api/experiments", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: `Strategy comparison · ${strategy} · ${testCase.query_id}`,
            config: { strategy, k: testCase.k || 5, query_id: testCase.query_id, mmr_diversity: testCase.mmr_diversity ?? 0.35 },
            metrics: result.metrics,
            notes: "Saved from the same-query TF-IDF vs hybrid vs MMR comparison."
          })
        })
      ));
      await loadExperiments();
      setMessage(`Comparison complete. Best for this query: ${data.best_strategy}. All three runs saved.`);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setDashboardBusy(false);
    }
  }

  async function runBatchBenchmark(event) {
    event.preventDefault();
    let dataset;
    try {
      dataset = JSON.parse(batchDataset);
    } catch {
      setMessage("Batch dataset must be valid JSON.");
      return;
    }
    setDashboardBusy(true);
    setMessage("");
    setBatchResult(null);
    try {
      const data = await apiRequest("/api/benchmark/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(dataset)
      });
      setBatchResult(data);
      await loadExperiments();
      setMessage(`Batch benchmark complete: ${data.case_count} cases across ${Object.keys(data.strategies).length} strategies. Runs saved.`);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setDashboardBusy(false);
    }
  }

  async function replayBenchmarkSnapshot() {
    if (!batchResult?.snapshot_id) return;
    setReplayBusy(true);
    setMessage("");
    try {
      const data = await apiRequest(`/api/benchmark/snapshots/${batchResult.snapshot_id}/replay`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strategies: Object.keys(batchResult.strategies) })
      });
      setBatchResult(data);
      await loadExperiments();
      setMessage("Snapshot replay completed using the saved corpus and labeled cases.");
    } catch (error) {
      setMessage(error.message);
    } finally {
      setReplayBusy(false);
    }
  }

  async function generateBenchmarkReport() {
    if (!reportDatasetName.trim()) return setMessage("Enter a dataset name for the report.");
    setDashboardBusy(true);
    setMessage("");
    setBenchmarkReport(null);
    try {
      const query = new URLSearchParams({ dataset_name: reportDatasetName.trim(), strategy: reportStrategy });
      const data = await apiRequest(`/api/benchmark/report?${query.toString()}`);
      setBenchmarkReport(data);
      setMessage(`Loaded ${data.run_count} historical benchmark run(s).`);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setDashboardBusy(false);
    }
  }

  async function generateBenchmarkInsights() {
    if (!reportDatasetName.trim()) return setMessage("Enter a dataset name for insights.");
    setDashboardBusy(true);
    setMessage("");
    setBenchmarkInsights(null);
    try {
      const query = new URLSearchParams({ dataset_name: reportDatasetName.trim() });
      const data = await apiRequest(`/api/benchmark/insights?${query.toString()}`);
      setBenchmarkInsights(data);
      setMessage(data.strategy_count ? "Benchmark intelligence generated." : "No comparable benchmark runs found yet.");
    } catch (error) {
      setMessage(error.message);
    } finally {
      setDashboardBusy(false);
    }
  }

  async function generateFailureAnalytics() {
    if (!reportDatasetName.trim()) return setMessage("Enter a dataset name for failure analytics.");
    setDashboardBusy(true);
    setMessage("");
    setFailureAnalytics(null);
    try {
      const query = new URLSearchParams({ dataset_name: reportDatasetName.trim() });
      const data = await apiRequest(`/api/benchmark/failure-analytics?${query.toString()}`);
      setFailureAnalytics(data);
      setMessage(`Failure analytics generated: ${data.signal_count} diagnostic signal(s).`);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setDashboardBusy(false);
    }
  }

  async function checkBenchmarkRegression(event) {
    event.preventDefault();
    setDashboardBusy(true);
    setMessage("");
    setRegressionResult(null);
    try {
      const data = await apiRequest("/api/benchmark/regression", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          baseline_run_id: Number(baselineRunId),
          current_run_id: Number(currentRunId),
          max_metric_drop: 0.05
        })
      });
      setRegressionResult(data);
      setMessage(data.status === "regression" ? "Regression detected. Review the metrics below." : "No quality regression detected within the configured tolerance.");
    } catch (error) {
      setMessage(error.message);
    } finally {
      setDashboardBusy(false);
    }
  }

  async function evaluateAnswerQuality() {
    if (!result?.answer || !result?.retrieved_chunks?.length) {
      setMessage("Run a knowledge-base query first so there is an answer and retrieved evidence to evaluate.");
      return;
    }
    setDashboardBusy(true);
    setMessage("");
    setQualityResult(null);
    try {
      let relevantIds;
      try {
        const parsed = JSON.parse(benchmarkCase);
        if (Array.isArray(parsed.relevant_chunk_ids)) relevantIds = parsed.relevant_chunk_ids;
      } catch {
        // Faithfulness judging does not require a labeled benchmark case.
      }
      const payload = {
        question: result.question,
        answer: result.answer,
        retrieved_chunk_ids: result.retrieved_chunks.map(chunk => chunk.chunk_id)
      };
      if (relevantIds) payload.relevant_chunk_ids = relevantIds;
      const data = await apiRequest("/api/quality/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      setQualityResult(data);
      await apiRequest("/api/experiments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: `Answer quality · ${result.question.slice(0, 48)}`,
          config: { method: "gemini-llm-as-judge", retrieved_chunk_ids: payload.retrieved_chunk_ids },
          metrics: { faithfulness_score: data.grounding.faithfulness_score, hallucination_detected: data.grounding.hallucination_detected, ...(data.context_metrics || {}) },
          notes: "Model-assisted answer grounding evaluation; inspect claim-level evidence."
        })
      });
      await loadExperiments();
      setMessage("Answer grounding evaluated and saved to experiment history.");
    } catch (error) {
      setMessage(error.message);
    } finally {
      setDashboardBusy(false);
    }
  }

  async function refreshDashboard() {
    setDashboardBusy(true);
    try {
      await Promise.all([loadExperiments(), loadDocuments(), loadTraces()]);
      setMessage("Dashboard data refreshed.");
    } catch (error) {
      setMessage(error.message);
    } finally {
      setDashboardBusy(false);
    }
  }

  async function loadIncidentHistory() {
    const params = new URLSearchParams();
    if (incidentPatternFilter) params.set("pattern_id", incidentPatternFilter);
    if (incidentSeverityFilter) params.set("severity", incidentSeverityFilter);
    params.set("limit", "100");
    const data = await apiRequest(`/api/diagnostics/incidents?${params.toString()}`);
    setIncidentHistory(data.incidents || []);
    setIncidentSummary(data.summary || { total_incidents: 0, patterns: [], severity: {} });
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
      const data = await apiRequest("/api/diagnostics/incident", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description: incident, logs: incidentLogs, expected_behavior: expectedBehavior })
      });
      setIncidentResult(data.diagnosis);
      await loadIncidentHistory();
      setMessage(`Diagnosis saved to incident history · ${data.incident_id.slice(0, 8)}`);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  const latestMetrics = experiments[0]?.metrics || {};
  const metricCards = [
    ["Recall@K", latestMetrics.recall_at_k],
    ["MRR", latestMetrics.mrr],
    ["nDCG@K", latestMetrics.ndcg_at_k],
    ["Evidence coverage", latestMetrics.evidence_term_coverage ?? latestMetrics.expected_answer_term_coverage]
  ];

  return (
    <main className="shell">
      <header className="hero">
        <div className="eyebrow">AI RELIABILITY WORKBENCH</div>
        <h1>RAG Failure <span>Diagnostics</span></h1>
        <p>Inspect retrieval quality, diagnose incidents, and evaluate your RAG pipeline.</p>
        <div className="status"><i /> Gemini-grounded answers · Incident triage · Evaluation lab</div>
      </header>

      {message && <div className="notice" role="status">{message}</div>}

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
          {result.retrieved_chunks.map(chunk => <article className="chunk" key={chunk.chunk_id}><div><b>{chunk.filename} · chunk #{chunk.chunk_id}</b><span>Score: {chunk.score}</span></div><p>{chunk.content}</p></article>)}
          <p className="hint">{result.diagnostics.note}</p>
        </div>}
      </section>

      <section className="panel">
        <div className="eyebrow">03 / INCIDENT TRIAGE CLINIC</div>
        <h2>Diagnose a RAG / LLM failure</h2>
        <p className="hint">Describe an incident. Gemini maps it to one primary failure pattern (P01–P12), identifies missing evidence, and suggests a minimal structural fix.</p>
        <form onSubmit={runIncidentDiagnosis}>
          <textarea value={incident} onChange={e => setIncident(e.target.value)} placeholder="Example: Retrieved chunks say refunds are not allowed, but the assistant confidently tells users they can get a refund..." rows="4" />
          <textarea value={expectedBehavior} onChange={e => setExpectedBehavior(e.target.value)} placeholder="Expected behavior (optional)" rows="2" />
          <textarea value={incidentLogs} onChange={e => setIncidentLogs(e.target.value)} placeholder="Relevant logs / traces (optional)" rows="3" />
          <button disabled={busy}>{busy ? "Diagnosing..." : "Run incident diagnosis"}</button>
        </form>
        {incidentResult && <div className="results">
          <div className="diagnostic"><div><div className="eyebrow">PRIMARY FAILURE PATTERN</div><strong>{incidentResult.primary_pattern_id} · {incidentResult.severity?.toUpperCase()}</strong></div><div className="score">{Math.round(incidentResult.confidence * 100)}%<small>model confidence</small></div></div>
          <h3>{incidentResult.primary_pattern_id} diagnosis</h3><p>{incidentResult.reasoning?.join(" ")}</p>
          {incidentResult.secondary_pattern_ids?.length > 0 && <p className="hint">Secondary candidates: {incidentResult.secondary_pattern_ids.join(", ")}</p>}
          <h3>Minimal structural fix</h3>{incidentResult.minimal_structural_fix?.map((item, i) => <p className="recommendation" key={i}>→ {item}</p>)}
          <h3>Missing evidence</h3>{incidentResult.missing_evidence?.map((item, i) => <p className="issue" key={i}>? {item}</p>)}
          <h3>Verification steps</h3>{incidentResult.verification_steps?.map((item, i) => <p className="recommendation" key={i}>✓ {item}</p>)}
          <p className="hint">{incidentResult.disclaimer}</p>
        </div>}
      <div className="subsection">
          <div className="eyebrow">PERSISTENT INCIDENT INTELLIGENCE</div>
          <h3>Incident history & pattern trends</h3>
          <div className="metric-grid">
            <div className="metric-card"><strong>{incidentSummary.total_incidents}</strong><span>Total saved incidents</span></div>
            {["critical", "high", "medium", "low"].map(level => <div className="metric-card" key={level}><strong>{incidentSummary.severity?.[level] || 0}</strong><span>{level} severity</span></div>)}
          </div>
          <div className="report-controls">
            <label>Pattern<select value={incidentPatternFilter} onChange={e => setIncidentPatternFilter(e.target.value)}>
              <option value="">All patterns</option>{Array.from({ length: 12 }, (_, index) => `P${String(index + 1).padStart(2, "0")}`).map(id => <option key={id} value={id}>{id}</option>)}
            </select></label>
            <label>Severity<select value={incidentSeverityFilter} onChange={e => setIncidentSeverityFilter(e.target.value)}>
              <option value="">All severities</option><option value="critical">Critical</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option>
            </select></label>
            <button className="secondary compact" onClick={() => loadIncidentHistory().catch(error => setMessage(error.message))}>Apply filters</button>
            <a className="report-download" href={`${API}/api/diagnostics/incidents.csv${incidentPatternFilter || incidentSeverityFilter ? `?${new URLSearchParams({ ...(incidentPatternFilter ? { pattern_id: incidentPatternFilter } : {}), ...(incidentSeverityFilter ? { severity: incidentSeverityFilter } : {}) }).toString()}` : ""}`}>Export CSV</a>
          </div>
          {incidentHistory.length === 0 ? <div className="empty-state">No saved incidents match these filters.</div> :
            <div className="table-wrap"><table className="experiment-table">
              <thead><tr><th>Incident</th><th>Pattern</th><th>Severity</th><th>Confidence</th><th>Created</th></tr></thead>
              <tbody>{incidentHistory.map(item => <tr key={item.incident_id}>
                <td><b>{item.description.slice(0, 100)}</b><small>{item.incident_id}</small></td>
                <td>{item.primary_pattern_id}<small>{item.secondary_pattern_ids.join(", ")}</small></td>
                <td>{item.severity}</td><td>{(item.confidence * 100).toFixed(0)}%</td><td>{item.created_at}</td>
              </tr>)}</tbody>
            </table></div>}
          <p className="hint">Pattern counts are based on saved primary diagnoses. LLM confidence is a model estimate, not a calibrated probability.</p>
        </div>

</section>

      <section className="panel" id="evaluation">
        <div className="section-heading"><div><div className="eyebrow">04 / EVALUATION LAB</div><h2>Baseline evaluation</h2></div><span className="count">Auto-save enabled</span></div>
        <p className="hint">Provide a reference answer. The baseline checks reference-answer term overlap in retrieved context.</p>
        <textarea value={expected} onChange={e => setExpected(e.target.value)} placeholder="Enter the expected answer..." rows="3" />
        <button className="secondary" onClick={runEvaluation} disabled={busy}>{busy ? "Evaluating..." : "Evaluate & save experiment"}</button>
        {evaluation && <div className="evaluation"><strong>{(evaluation.evaluation.expected_answer_term_coverage * 100).toFixed(1)}%</strong><span>Expected-answer term coverage</span><p>{evaluation.evaluation.interpretation}</p></div>}
        <div className="subsection">
          <div className="eyebrow">ANSWER GROUNDING EVALUATION</div>
          <h3>Faithfulness & hallucination check</h3>
          <p className="hint">Run a knowledge-base query above first. Gemini will inspect the generated answer against its retrieved chunks. Optionally, the labeled benchmark IDs are used for context precision and recall.</p>
          <button className="secondary" onClick={evaluateAnswerQuality} disabled={dashboardBusy || !result}>
            {dashboardBusy ? "Evaluating..." : "Evaluate answer faithfulness"}
          </button>
          {qualityResult && <div className="results">
            <div className="diagnostic">
              <div><div className="eyebrow">FAITHFULNESS · {qualityResult.grounding.judge}</div><strong>{(qualityResult.grounding.faithfulness_score * 100).toFixed(1)}%</strong></div>
              <div className="score">{qualityResult.grounding.hallucination_detected ? "Detected" : "None"}<small>unsupported claims</small></div>
            </div>
            {qualityResult.grounding.unsupported_claims.length > 0 && <>
              <h3>Unsupported claims</h3>
              {qualityResult.grounding.unsupported_claims.map((claim, i) => <p className="issue" key={i}>⚠ {claim}</p>)}
            </>}
            {qualityResult.grounding.supported_claims.length > 0 && <>
              <h3>Supported claims</h3>
              {qualityResult.grounding.supported_claims.map((claim, i) => <p className="recommendation" key={i}>✓ {claim}</p>)}
            </>}
            {qualityResult.context_metrics && <div className="metric-grid">
              <div className="metric-card"><strong>{qualityResult.context_metrics.context_precision.toFixed(3)}</strong><span>Context precision</span></div>
              <div className="metric-card"><strong>{qualityResult.context_metrics.context_recall.toFixed(3)}</strong><span>Context recall</span></div>
            </div>}
            <p className="hint">{qualityResult.grounding.limitations}</p>
          </div>}
        </div>
        <div className="subsection">
          <div className="eyebrow">LABELED RETRIEVAL BENCHMARK</div>
          <h3>Run Recall@K · MRR · nDCG@K</h3>
          <p className="hint">Paste a benchmark case as JSON. The relevant_chunk_ids must be real chunk IDs from your indexed documents. The starter sample is illustrative only.</p>
          <form onSubmit={runBenchmark}>
            <textarea className="code-input" value={benchmarkCase} onChange={e => setBenchmarkCase(e.target.value)} rows="9" spellCheck="false" />
            <button disabled={dashboardBusy}>{dashboardBusy ? "Running benchmark..." : "Run labeled benchmark"}</button>
          </form>
          {benchmarkResult && <div className="metric-grid">
            {Object.entries(benchmarkResult.metrics).filter(([key, value]) => typeof value === "number" && key !== "retrieved_count").map(([key, value]) => <div className="metric-card" key={key}><strong>{Number(value).toFixed(3)}</strong><span>{key.replaceAll("_", " ")}</span></div>)}
          </div>}
          <button className="secondary" onClick={compareStrategies} disabled={dashboardBusy}>
            {dashboardBusy ? "Comparing..." : "Compare TF-IDF vs Hybrid vs MMR"}
          </button>
          {comparisonResult && <div className="subsection">
            <div className="diagnostic">
              <div><div className="eyebrow">BEST STRATEGY FOR THIS QUERY</div><strong>{comparisonResult.best_strategy.replaceAll("_", " ")}</strong></div>
            </div>
            <div className="table-wrap"><table className="experiment-table">
              <thead><tr><th>Strategy</th><th>Recall@K</th><th>MRR</th><th>nDCG@K</th><th>Evidence coverage</th></tr></thead>
              <tbody>{Object.entries(comparisonResult.comparison).map(([strategy, row]) => <tr key={strategy}>
                <td><b>{strategy.replaceAll("_", " ")}</b>{strategy === comparisonResult.best_strategy && <small>Best for this query</small>}</td>
                <td>{row.metrics.recall_at_k.toFixed(3)}</td><td>{row.metrics.mrr.toFixed(3)}</td>
                <td>{row.metrics.ndcg_at_k.toFixed(3)}</td><td>{row.metrics.evidence_term_coverage.toFixed(3)}</td>
              </tr>)}</tbody>
            </table></div>
            <p className="hint">{comparisonResult.selection_note} Results are valid only for this labeled query and current indexed corpus.</p>
          </div>}
          <div className="subsection">
            <div className="eyebrow">BATCH REGRESSION BENCHMARK</div>
            <h3>Evaluate a dataset across retrieval strategies</h3>
            <p className="hint">Run 1–100 labeled questions in one request. Use real integer chunk IDs from the currently indexed corpus. Aggregate timing is an in-process measurement, not a load test.</p>
            <form onSubmit={runBatchBenchmark}>
              <textarea className="code-input" value={batchDataset} onChange={e => setBatchDataset(e.target.value)} rows="12" spellCheck="false" />
              <button disabled={dashboardBusy}>{dashboardBusy ? "Running dataset..." : "Run batch benchmark"}</button>
            </form>
            {batchResult && <>
              <p className="hint">{batchResult.dataset_name} · {batchResult.case_count} labeled cases</p>
              {Object.entries(batchResult.strategies).map(([strategy, row]) => <div className="results" key={strategy}>
                <h3>{strategy.replaceAll("_", " ")}</h3>
                <div className="metric-grid">
                  {["recall_at_k", "mrr", "ndcg_at_k", "evidence_term_coverage", "latency_avg_ms", "latency_p50_ms", "latency_p95_ms"].map(key =>
                    <div className="metric-card" key={key}><strong>{Number(row.metrics[key] || 0).toFixed(3)}</strong><span>{key.replaceAll("_", " ")}</span></div>
                  )}
                </div>
                <p className="hint">Experiment #{row.experiment_id}</p>
              </div>)}
              <p className="hint">Snapshot #{batchResult.snapshot_id} · corpus {batchResult.corpus_fingerprint?.slice(0, 12)}… · dataset {batchResult.dataset_fingerprint?.slice(0, 12)}…</p>
              <button className="secondary compact" onClick={replayBenchmarkSnapshot} disabled={dashboardBusy || replayBusy}>{replayBusy ? "Replaying snapshot..." : "Replay saved snapshot"}</button>
              <p className="hint">{batchResult.metric_note}</p>
            </>}
          </div>
          <p className="hint">Ranking metrics require manually labeled relevant chunks. Evidence-term coverage is lexical—not a semantic faithfulness score.</p>
        </div>
      </section>

      <section className="panel" id="experiments">
        <div className="section-heading"><div><div className="eyebrow">05 / EXPERIMENT TRACKING</div><h2>Experiment dashboard</h2></div><button className="secondary compact" onClick={refreshDashboard} disabled={dashboardBusy}>{dashboardBusy ? "Refreshing..." : "Refresh"}</button></div>
        <div className="metric-grid">
          <div className="metric-card"><strong>{experiments.length}</strong><span>Saved runs shown</span></div>
          <div className="metric-card"><strong>{documents.length}</strong><span>Indexed documents</span></div>
          {metricCards.map(([label, value]) => <div className="metric-card" key={label}><strong>{typeof value === "number" ? value.toFixed(3) : "—"}</strong><span>Latest · {label}</span></div>)}
        </div>
        <p className="hint">Latest metrics are taken from the most recent saved run. A dash means that metric was not part of that run.</p>
        <div className="subsection">
          <div className="eyebrow">HISTORICAL BENCHMARK REPORT</div>
          <h3>Metric trends & export</h3>
          <p className="hint">Enter the exact dataset name used in batch runs. The report shows saved history for one retrieval strategy.</p>
          <div className="report-controls">
            <label>Dataset name<input value={reportDatasetName} onChange={e => setReportDatasetName(e.target.value)} placeholder="Exact dataset name" /></label>
            <label>Strategy<select value={reportStrategy} onChange={e => setReportStrategy(e.target.value)}>
              <option value="tfidf">TF-IDF</option><option value="hybrid">Hybrid</option><option value="hybrid_mmr">Hybrid MMR</option>
            </select></label>
            <button onClick={generateBenchmarkReport} disabled={dashboardBusy}>{dashboardBusy ? "Generating..." : "Generate report"}</button>
          </div>
          {benchmarkReport && <>
            <div className="metric-grid">
              {Object.entries(benchmarkReport.trends).map(([metric, trend]) => <div className="metric-card" key={metric}>
                <strong>{trend.latest === null ? "—" : trend.latest.toFixed(3)}</strong>
                <span>{metric.replaceAll("_", " ")} · first-to-latest {trend.delta > 0 ? "+" : ""}{trend.delta.toFixed(3)}</span>
              </div>)}
            </div>
            <p className="hint">{benchmarkReport.run_count} runs · {benchmarkReport.dataset_name} · {benchmarkReport.strategy}</p>
            <div className="table-wrap"><table className="experiment-table">
              <thead><tr><th>Run</th><th>Date</th><th>Recall@K</th><th>MRR</th><th>nDCG@K</th><th>Avg latency</th></tr></thead>
              <tbody>{benchmarkReport.history.map(run => <tr key={run.run_id}>
                <td>#{run.run_id}</td><td>{run.created_at}</td><td>{run.metrics.recall_at_k.toFixed(3)}</td>
                <td>{run.metrics.mrr.toFixed(3)}</td><td>{run.metrics.ndcg_at_k.toFixed(3)}</td><td>{run.metrics.latency_avg_ms.toFixed(2)} ms</td>
              </tr>)}</tbody>
            </table></div>
            <a className="report-download" href={`${API}/api/benchmark/report.csv?dataset_name=${encodeURIComponent(reportDatasetName.trim())}&strategy=${encodeURIComponent(reportStrategy)}`}>Download CSV report</a>
            <p className="hint">{benchmarkReport.note}</p>
          </>}
        </div>
<div className="subsection">
          <div className="eyebrow">BENCHMARK INTELLIGENCE</div>
          <h3>Strategy scorecards & recommendations</h3>
          <p className="hint">Analyze the latest TF-IDF, Hybrid, and Hybrid MMR runs for this dataset. Recommendations are only produced when latest runs share the same snapshot fingerprints and case count.</p>
          <button className="secondary compact" onClick={generateBenchmarkInsights} disabled={dashboardBusy}>{dashboardBusy ? "Analyzing..." : "Generate benchmark insights"}</button>
          {benchmarkInsights && <>
            {benchmarkInsights.best_strategy && <div className="diagnostic">
              <div><div className="eyebrow">HIGHEST QUALITY SCORE</div><strong>{benchmarkInsights.best_strategy.replaceAll("_", " ").toUpperCase()}</strong></div>
              <div className="score">{(benchmarkInsights.strategies.find(row => row.strategy === benchmarkInsights.best_strategy)?.quality_score ?? 0).toFixed(3)}<small>mean quality score</small></div>
            </div>}
            {!benchmarkInsights.comparable && <p className="issue">Strategy runs are not comparable yet. Run all strategies on the same saved snapshot.</p>}
            {benchmarkInsights.strategies.length > 0 && <div className="table-wrap"><table className="experiment-table">
              <thead><tr><th>Strategy</th><th>Quality score</th><th>Recall@K</th><th>MRR</th><th>nDCG@K</th><th>Evidence</th><th>p95 latency</th></tr></thead>
              <tbody>{benchmarkInsights.strategies.map(row => <tr key={row.strategy}>
                <td>{row.strategy.replaceAll("_", " ")}</td><td>{row.quality_score.toFixed(3)}</td>
                <td>{row.metrics.recall_at_k.toFixed(3)}</td><td>{row.metrics.mrr.toFixed(3)}</td>
                <td>{row.metrics.ndcg_at_k.toFixed(3)}</td><td>{row.metrics.evidence_term_coverage.toFixed(3)}</td>
                <td>{row.metrics.latency_p95_ms.toFixed(2)} ms</td>
              </tr>)}</tbody>
            </table></div>}
            <h4>Recommendations</h4>
            {benchmarkInsights.recommendations.map((item, index) => <p className="recommendation" key={index}>→ {item}</p>)}
            <p className="hint">{benchmarkInsights.note}</p>
          </>}
        </div>
<div className="subsection">
          <div className="eyebrow">RAG FAILURE ANALYTICS</div>
          <h3>Failure pattern distribution</h3>
          <p className="hint">Generate rule-based signals from the latest saved run per retrieval strategy.</p>
          <button className="secondary compact" onClick={generateFailureAnalytics} disabled={dashboardBusy}>{dashboardBusy ? "Analyzing..." : "Analyze failure patterns"}</button>
          {failureAnalytics && <>
            <div className="metric-grid">
              <div className="metric-card"><strong>{failureAnalytics.analyzed_strategy_count}</strong><span>Strategies analyzed</span></div>
              <div className="metric-card"><strong>{failureAnalytics.signal_count}</strong><span>Diagnostic signals</span></div>
              <div className="metric-card"><strong>{failureAnalytics.distribution.length}</strong><span>Patterns detected</span></div>
            </div>
            {failureAnalytics.distribution.length > 0 ? <div className="table-wrap"><table className="experiment-table">
              <thead><tr><th>Pattern</th><th>Failure type</th><th>Signals</th></tr></thead>
              <tbody>{failureAnalytics.distribution.map(row => <tr key={row.pattern_id}>
                <td>{row.pattern_id}</td><td>{row.name}<small>{row.symptom}</small></td><td>{row.signal_count}</td>
              </tr>)}</tbody>
            </table></div> : <p className="hint">No rule-based failure signals were detected in the latest saved runs.</p>}
            <h4>Evidence & recommended fixes</h4>
            {failureAnalytics.signals.map((signal, index) => <div className="results" key={`${signal.run_id}-${signal.pattern_id}-${index}`}>
              <strong>{signal.pattern_id} · {signal.severity.toUpperCase()} · {signal.strategy}</strong>
              <p>{signal.evidence}</p><p className="recommendation">→ {signal.recommendation}</p>
              <p className="hint">Experiment #{signal.run_id} · Snapshot #{signal.snapshot_id ?? "not recorded"}</p>
            </div>)}
            <p className="hint">{failureAnalytics.note}</p>
          </>}
        </div>

<div className="subsection">
          <div className="eyebrow">AUTOMATED QUALITY GATE</div>
          <h3>Baseline vs current benchmark</h3>
          <p className="hint">Compare two saved batch runs from the same dataset, strategy and case count. A quality metric is flagged when it drops by more than 0.05 (5 percentage points).</p>
          <form onSubmit={checkBenchmarkRegression}>
            <div className="regression-inputs">
              <label>Baseline experiment ID<input type="number" min="1" value={baselineRunId} onChange={e => setBaselineRunId(e.target.value)} placeholder="e.g. 12" required /></label>
              <label>Current experiment ID<input type="number" min="1" value={currentRunId} onChange={e => setCurrentRunId(e.target.value)} placeholder="e.g. 15" required /></label>
            </div>
            <button disabled={dashboardBusy || !baselineRunId || !currentRunId}>{dashboardBusy ? "Checking..." : "Check for regression"}</button>
          </form>
          {regressionResult && <div className="results">
            <div className="diagnostic">
              <div><div className="eyebrow">REGRESSION GATE</div><strong>{regressionResult.status === "passed" ? "No regression detected" : "Quality regression detected"}</strong></div>
              <div className="score">{regressionResult.status === "passed" ? "PASS" : "FAIL"}<small>{regressionResult.strategy} · {regressionResult.dataset_name}</small></div>
            </div>
            <div className="table-wrap"><table className="experiment-table">
              <thead><tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Delta</th><th>Gate</th></tr></thead>
              <tbody>{Object.entries(regressionResult.comparisons).map(([metric, row]) => <tr key={metric}>
                <td>{metric.replaceAll("_", " ")}</td><td>{row.baseline.toFixed(3)}</td><td>{row.current.toFixed(3)}</td>
                <td>{row.delta > 0 ? "+" : ""}{row.delta.toFixed(3)}</td><td>{row.regression ? "REGRESSION" : "PASS"}</td>
              </tr>)}</tbody>
            </table></div>
            <p className="hint">{regressionResult.note}</p>
          </div>}
        </div>

        {experiments.length === 0 ? <div className="empty-state">No experiments saved yet. Run a baseline evaluation or labeled benchmark to create your first run.</div> :
          <div className="table-wrap"><table className="experiment-table"><thead><tr><th>Run</th><th>Configuration</th><th>Metrics</th><th>Created</th></tr></thead><tbody>
            {experiments.map(run => <tr key={run.id}><td><b>#{run.id} · {run.name}</b>{run.notes && <small>{run.notes}</small>}</td><td><code>{JSON.stringify(run.config)}</code></td><td><code>{JSON.stringify(run.metrics)}</code></td><td>{run.created_at}</td></tr>)}
          </tbody></table></div>}
      </section>
      <section className="panel" id="observability">
        <div className="section-heading"><div><div className="eyebrow">06 / OBSERVABILITY & TRACING</div><h2>Query traces</h2></div><button className="secondary compact" onClick={loadTraces}>Refresh traces</button></div>
        <p className="hint">Har query ka unique trace ID, embedding/retrieval/generation latency aur total response time yahan store hota hai. Token usage abhi unavailable hai jab tak provider response metadata expose nahi karta; cost estimate intentionally show nahi kiya gaya.</p>
        {traces.length === 0 ? <div className="empty-state">No query traces yet. Run a knowledge-base query to create the first trace.</div> :
          <div className="table-wrap"><table className="experiment-table"><thead><tr><th>Trace</th><th>Mode</th><th>Embedding</th><th>Retrieval</th><th>Generation</th><th>Total</th><th>Time</th></tr></thead><tbody>
            {traces.map(trace => <tr key={trace.trace_id} onClick={() => inspectTrace(trace.trace_id)} className="trace-row">
              <td><button className="trace-link" onClick={() => inspectTrace(trace.trace_id)}>{trace.trace_id.slice(0, 8)}…</button><small>{trace.question}</small></td>
              <td>{trace.answer_mode} / {trace.retrieval_mode}</td><td>{trace.timings.embedding_ms} ms</td><td>{trace.timings.retrieval_ms} ms</td><td>{trace.timings.generation_ms} ms</td><td><b>{trace.timings.total_ms} ms</b></td><td>{trace.created_at}</td>
            </tr>)}
          </tbody></table></div>}
        {selectedTrace && <div className="answer"><div className="eyebrow">TRACE DETAIL · {selectedTrace.trace_id}</div>
          <p><b>Question:</b> {selectedTrace.question}</p><p><b>Retrieved chunks:</b> {selectedTrace.retrieved_count} · <b>Top K:</b> {selectedTrace.top_k}</p>
          <p><b>Token usage:</b> {selectedTrace.token_usage ? JSON.stringify(selectedTrace.token_usage) : "Not reported by provider"}</p>
          {selectedTrace.error && <p className="issue">Error: {selectedTrace.error}</p>}
          <button className="secondary compact" onClick={() => setSelectedTrace(null)}>Close details</button>
        </div>}
      </section>
      <footer>RAG Failure Diagnostics · Gemini grounded generation · Incident clinic · Evaluation lab</footer>
    </main>
  );
}
