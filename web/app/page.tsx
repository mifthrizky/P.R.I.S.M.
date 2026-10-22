"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";

type Metrics = {
  air_temperature_k: number;
  process_temperature_k: number;
  rotational_speed_rpm: number;
  torque_nm: number;
  tool_wear_min: number;
};

type Citation = { document_id: string; title: string; page: number };

type Prediction = {
  request_id: string;
  status: "HEALTHY" | "CRITICAL_WARNING";
  failure_probability: number;
  threshold: number;
  primary_failure: string | null;
  failure_predictions: Array<{ failure_type: string; probability: number }>;
  machine_type: string;
  machine_type_assumed: boolean;
  model_version: string;
  mitigation: null | {
    status: "MITIGATION_READY" | "MANUAL_REVIEW_REQUIRED";
    reason: string | null;
    mitigation_plan: null | {
      immediate_action: string;
      required_ppe: string[];
      safety_steps: string[];
      citations: Citation[];
    };
  };
  timings: {
    ml_ms: number;
    retrieval_ms: number | null;
    generation_ms: number | null;
    total_ms: number;
  };
};

const presets = {
  normal: {
    air_temperature_k: 298.1,
    process_temperature_k: 308.6,
    rotational_speed_rpm: 1551,
    torque_nm: 42.8,
    tool_wear_min: 21,
  },
  anomaly: {
    air_temperature_k: 304.5,
    process_temperature_k: 313.2,
    rotational_speed_rpm: 1201,
    torque_nm: 71.2,
    tool_wear_min: 215,
  },
} satisfies Record<string, Metrics>;

const metricFields: Array<{ key: keyof Metrics; label: string; unit: string; step: number }> = [
  { key: "air_temperature_k", label: "Air temperature", unit: "K", step: 0.1 },
  { key: "process_temperature_k", label: "Process temperature", unit: "K", step: 0.1 },
  { key: "rotational_speed_rpm", label: "Rotational speed", unit: "rpm", step: 1 },
  { key: "torque_nm", label: "Torque", unit: "Nm", step: 0.1 },
  { key: "tool_wear_min", label: "Tool wear", unit: "min", step: 1 },
];

function formatMs(value: number | null) {
  return value === null ? "Skipped" : `${value.toFixed(1)} ms`;
}

export default function Dashboard() {
  const [metrics, setMetrics] = useState<Metrics>(presets.normal);
  const [machineType, setMachineType] = useState("M");
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState<boolean | null>(null);

  const checkHealth = useCallback(async () => {
    try {
      const response = await fetch("/api/health", { cache: "no-store" });
      setReady(response.ok);
    } catch {
      setReady(false);
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(checkHealth, 0);
    const timer = window.setInterval(checkHealth, 15_000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [checkHealth]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          sensor_id: "CNC-Milling-01",
          timestamp: new Date().toISOString(),
          machine_type: machineType,
          metrics,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(typeof payload.detail === "string" ? payload.detail : "Request failed");
      }
      setResult(payload);
      setReady(true);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to run prediction");
    } finally {
      setLoading(false);
    }
  }

  const critical = result?.status === "CRITICAL_WARNING";
  const risk = result ? Math.round(result.failure_probability * 100) : 0;

  return (
    <main>
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true">P</div>
        <div>
          <p className="eyebrow">Predictive Retrieval for Industrial Safety</p>
          <h1>PRISM <span>Operations</span></h1>
        </div>
        <div className={`system-state ${ready ? "online" : ready === false ? "offline" : "checking"}`}>
          <i aria-hidden="true" />
          {ready ? "System ready" : ready === false ? "System offline" : "Checking system"}
        </div>
      </header>

      <section className="hero">
        <div>
          <p className="section-index">CONTROL ROOM / 01</p>
          <h2>Machine intelligence,<br /><em>grounded in safety.</em></h2>
        </div>
        <p className="hero-copy">
          Evaluate live sensor conditions. Healthy machines bypass generation; anomalies
          trigger evidence retrieval and validated local mitigation.
        </p>
      </section>

      <div className="workspace">
        <form className="panel input-panel" onSubmit={submit}>
          <div className="panel-heading">
            <div>
              <p className="section-index">SENSOR INPUT / 02</p>
              <h3>CNC-Milling-01</h3>
            </div>
            <select value={machineType} onChange={(event) => setMachineType(event.target.value)} aria-label="Machine type">
              <option value="L">Type L</option>
              <option value="M">Type M</option>
              <option value="H">Type H</option>
            </select>
          </div>

          <div className="preset-row" aria-label="Sensor presets">
            <button type="button" onClick={() => setMetrics(presets.normal)}>Load normal</button>
            <button type="button" onClick={() => setMetrics(presets.anomaly)}>Load anomaly</button>
          </div>

          <div className="metrics-grid">
            {metricFields.map((field) => (
              <label key={field.key}>
                <span>{field.label}</span>
                <div className="input-wrap">
                  <input
                    type="number"
                    required
                    min="0"
                    step={field.step}
                    value={metrics[field.key]}
                    onChange={(event) =>
                      setMetrics((current) => ({
                        ...current,
                        [field.key]: Number(event.target.value),
                      }))
                    }
                  />
                  <b>{field.unit}</b>
                </div>
              </label>
            ))}
          </div>

          <button className="run-button" type="submit" disabled={loading}>
            <span>{loading ? "Running pipeline…" : "Run diagnostic"}</span>
            <span aria-hidden="true">↗</span>
          </button>
          {error && <p className="error-message" role="alert">{error}</p>}
        </form>

        <section className={`panel result-panel ${critical ? "critical" : ""}`} aria-live="polite">
          <div className="panel-heading">
            <div>
              <p className="section-index">DIAGNOSTIC / 03</p>
              <h3>{result ? (critical ? "Attention required" : "Operating normally") : "Awaiting signal"}</h3>
            </div>
            {result && <span className="status-badge">{result.status.replace("_", " ")}</span>}
          </div>

          {!result ? (
            <div className="empty-state">
              <div className="radar" aria-hidden="true"><i /></div>
              <p>Submit sensor metrics to begin the predictive maintenance pipeline.</p>
            </div>
          ) : (
            <>
              <div className="risk-block">
                <div className="risk-number"><strong>{risk}</strong><span>%</span></div>
                <div>
                  <span>Failure probability</span>
                  <div className="risk-track"><i style={{ width: `${Math.max(risk, 1)}%` }} /></div>
                  <small>Decision threshold {(result.threshold * 100).toFixed(2)}%</small>
                </div>
              </div>

              <div className="result-grid">
                <article>
                  <span>Primary failure</span>
                  <strong>{result.primary_failure ?? "None detected"}</strong>
                </article>
                <article>
                  <span>Model version</span>
                  <strong>{result.model_version}</strong>
                </article>
                <article>
                  <span>Total latency</span>
                  <strong>{formatMs(result.timings.total_ms)}</strong>
                </article>
              </div>

              <div className="timing-strip">
                <span>ML <b>{formatMs(result.timings.ml_ms)}</b></span>
                <span>Retrieval <b>{formatMs(result.timings.retrieval_ms)}</b></span>
                <span>Generation <b>{formatMs(result.timings.generation_ms)}</b></span>
              </div>
            </>
          )}
        </section>
      </div>

      {result?.mitigation && (
        <section className="mitigation-section">
          <div className="mitigation-title">
            <div>
              <p className="section-index">MITIGATION / 04</p>
              <h2>{result.mitigation.status === "MITIGATION_READY" ? "Grounded response" : "Manual review required"}</h2>
            </div>
            <code>{result.request_id}</code>
          </div>

          {result.mitigation.mitigation_plan ? (
            <div className="mitigation-grid">
              <article className="action-card">
                <span>Immediate action</span>
                <p>{result.mitigation.mitigation_plan.immediate_action}</p>
              </article>
              <article>
                <span>Required PPE</span>
                <ul>{result.mitigation.mitigation_plan.required_ppe.map((item) => <li key={item}>{item}</li>)}</ul>
              </article>
              <article>
                <span>Safety sequence</span>
                <ol>{result.mitigation.mitigation_plan.safety_steps.map((item) => <li key={item}>{item}</li>)}</ol>
              </article>
              <article>
                <span>Verified sources</span>
                <ul className="citations">
                  {result.mitigation.mitigation_plan.citations.map((citation) => (
                    <li key={`${citation.document_id}-${citation.page}`}>
                      <b>{citation.document_id}</b> · p. {citation.page}<small>{citation.title}</small>
                    </li>
                  ))}
                </ul>
              </article>
            </div>
          ) : (
            <p className="manual-review">{result.mitigation.reason}</p>
          )}
        </section>
      )}

      <footer>
        <span>PRISM v0.1.0</span>
        <p>Decision support only · Human authorization required before machine intervention</p>
      </footer>
    </main>
  );
}
