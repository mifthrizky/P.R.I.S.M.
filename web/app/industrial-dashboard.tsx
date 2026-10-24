"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

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
  sensor_id: string;
  timestamp: string;
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

type PresetName = "normal" | "anomaly" | "custom";

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
} satisfies Record<Exclude<PresetName, "custom">, Metrics>;

const metricFields = [
  { key: "air_temperature_k", label: "Air temp", unit: "K", step: 0.1, min: 150, max: 1000 },
  { key: "process_temperature_k", label: "Process temp", unit: "K", step: 0.1, min: 150, max: 1500 },
  { key: "rotational_speed_rpm", label: "Spindle speed", unit: "rpm", step: 1, min: 0, max: 100000 },
  { key: "torque_nm", label: "Torque", unit: "Nm", step: 0.1, min: 0, max: 100000 },
  { key: "tool_wear_min", label: "Tool wear", unit: "min", step: 1, min: 0, max: 1000000 },
] satisfies Array<{
  key: keyof Metrics;
  label: string;
  unit: string;
  step: number;
  min: number;
  max: number;
}>;

const sensorPattern = /^[A-Za-z0-9._:-]+$/;

function formatMs(value: number | null) {
  return value === null ? "BYPASSED" : `${value.toFixed(1)} ms`;
}

function formatTimestamp(value?: string) {
  if (!value) return "NO SAMPLE";
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(new Date(value));
}

function getApiError(payload: unknown) {
  if (!payload || typeof payload !== "object" || !("detail" in payload)) {
    return "The diagnostic service did not return a valid response.";
  }
  const detail = (payload as { detail: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) =>
        item && typeof item === "object" && "msg" in item ? String(item.msg) : "Invalid input",
      )
      .join(" · ");
  }
  return "The diagnostic request was rejected.";
}

export default function IndustrialDashboard() {
  const [metrics, setMetrics] = useState<Metrics>(presets.normal);
  const [activePreset, setActivePreset] = useState<PresetName>("normal");
  const [sensorId, setSensorId] = useState("CNC-Milling-01");
  const [machineType, setMachineType] = useState("M");
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState<boolean | null>(null);
  const [checkingConnection, setCheckingConnection] = useState(true);

  const validation = useMemo(() => {
    const errors: Partial<Record<keyof Metrics | "sensor_id", string>> = {};
    const cleanSensorId = sensorId.trim();
    if (!cleanSensorId) {
      errors.sensor_id = "Asset ID is required.";
    } else if (cleanSensorId.length > 128 || !sensorPattern.test(cleanSensorId)) {
      errors.sensor_id = "Use letters, numbers, dots, colons, underscores, or hyphens.";
    }
    metricFields.forEach((field) => {
      const value = metrics[field.key];
      if (!Number.isFinite(value) || value < field.min || value > field.max) {
        errors[field.key] = `Range ${field.min.toLocaleString()}–${field.max.toLocaleString()} ${field.unit}`;
      }
    });
    return errors;
  }, [metrics, sensorId]);

  const formValid = Object.keys(validation).length === 0;

  const checkHealth = useCallback(async (silent = false) => {
    if (!silent) setCheckingConnection(true);
    try {
      const response = await fetch("/api/health", { cache: "no-store" });
      setReady(response.ok);
    } catch {
      setReady(false);
    } finally {
      if (!silent) setCheckingConnection(false);
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(checkHealth, 0);
    const timer = window.setInterval(() => checkHealth(true), 15_000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [checkHealth]);

  function loadPreset(name: Exclude<PresetName, "custom">) {
    setMetrics(presets[name]);
    setActivePreset(name);
    setError(null);
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!formValid || ready !== true) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          sensor_id: sensorId.trim(),
          timestamp: new Date().toISOString(),
          machine_type: machineType,
          metrics,
        }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) throw new Error(getApiError(payload));
      setResult(payload as Prediction);
      setReady(true);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to run diagnostic.");
    } finally {
      setLoading(false);
    }
  }

  const critical = result?.status === "CRITICAL_WARNING";
  const risk = result ? result.failure_probability * 100 : 0;
  const threshold = result ? result.threshold * 100 : 0;
  const rankedFailures = result
    ? [...result.failure_predictions].sort((a, b) => b.probability - a.probability)
    : [];
  const consoleState = loading
    ? "processing"
    : error
      ? "fault"
      : ready === false
        ? "offline"
        : critical
          ? "critical"
          : result
            ? "healthy"
            : "standby";
  const statusCopy = {
    processing: ["PIPELINE ACTIVE", "Awaiting verified response"],
    fault: ["REQUEST FAULT", error ?? "Diagnostic interrupted"],
    offline: ["LINK OFFLINE", "Prediction service is unavailable"],
    critical: ["ACTION REQUIRED", `${result?.primary_failure ?? "Failure"} signature detected`],
    healthy: ["SYSTEM NOMINAL", "No failure signature above threshold"],
    standby: ["SYSTEM STANDBY", "Load sensor values and run a diagnostic"],
  } as const;

  return (
    <>
      <a className="skip-link" href="#main-console">Skip to diagnostic console</a>
      <header className="command-bar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><span>P</span></div>
          <div><p>Predictive safety system</p><h1>PRISM <span>Operations</span></h1></div>
        </div>
        <div className="command-context" aria-label="Current equipment context">
          <span>Active asset</span><strong>{sensorId.trim() || "UNASSIGNED"}</strong>
        </div>
        <button
          className={`connection-state ${checkingConnection ? "checking" : ready ? "online" : "offline"}`}
          type="button"
          onClick={() => checkHealth()}
          aria-label="Check prediction service connection"
        >
          <i aria-hidden="true" />
          <span>{checkingConnection ? "Checking link" : ready ? "Service ready" : "Service offline"}</span>
          <b aria-hidden="true">↻</b>
        </button>
      </header>

      <main id="main-console" className="console-shell">
        <section className="equipment-strip" aria-label="Equipment summary">
          <div><span>Operating mode</span><strong>Decision support</strong></div>
          <div><span>Machine class</span><strong>TYPE {machineType}</strong></div>
          <div><span>Model</span><strong>{result?.model_version ?? "AWAITING RUN"}</strong></div>
          <div><span>Last sample</span><strong>{formatTimestamp(result?.timestamp)}</strong></div>
        </section>

        <div className="workspace">
          <form className="control-panel panel" onSubmit={submit} noValidate>
            <div className="panel-title">
              <div><p className="kicker"><span>01</span> Sensor control</p><h2>Input channel</h2></div>
              <span className="channel-id">CH-A01</span>
            </div>

            <div className="identity-grid">
              <label className={validation.sensor_id ? "field-invalid" : ""}>
                <span>Asset ID</span>
                <input
                  type="text"
                  value={sensorId}
                  maxLength={128}
                  pattern="[A-Za-z0-9._:-]+"
                  aria-invalid={Boolean(validation.sensor_id)}
                  aria-describedby={validation.sensor_id ? "sensor-error" : undefined}
                  onChange={(event) => setSensorId(event.target.value)}
                />
                {validation.sensor_id && <small id="sensor-error">{validation.sensor_id}</small>}
              </label>
              <label>
                <span>Machine type</span>
                <select value={machineType} onChange={(event) => setMachineType(event.target.value)}>
                  <option value="L">Type L</option><option value="M">Type M</option><option value="H">Type H</option>
                </select>
              </label>
            </div>

            <fieldset className="preset-control">
              <legend>Test profile</legend>
              <div>
                <button type="button" className={activePreset === "normal" ? "active" : ""} aria-pressed={activePreset === "normal"} onClick={() => loadPreset("normal")}>Nominal</button>
                <button type="button" className={activePreset === "anomaly" ? "active alert" : ""} aria-pressed={activePreset === "anomaly"} onClick={() => loadPreset("anomaly")}>Anomaly</button>
                <span className={activePreset === "custom" ? "custom-active" : ""}>Custom</span>
              </div>
            </fieldset>

            <div className="metrics-grid">
              {metricFields.map((field) => (
                <label key={field.key} className={validation[field.key] ? "field-invalid" : ""}>
                  <span>{field.label}</span>
                  <div className="input-wrap">
                    <input
                      type="number"
                      required
                      min={field.min}
                      max={field.max}
                      step={field.step}
                      value={metrics[field.key]}
                      aria-invalid={Boolean(validation[field.key])}
                      onChange={(event) => {
                        setActivePreset("custom");
                        setMetrics((current) => ({
                          ...current,
                          [field.key]: event.target.value === "" ? Number.NaN : Number(event.target.value),
                        }));
                      }}
                    />
                    <b>{field.unit}</b>
                  </div>
                  {validation[field.key] && <small>{validation[field.key]}</small>}
                </label>
              ))}
            </div>

            <button className="run-button" type="submit" disabled={loading || ready !== true || !formValid}>
              <span>{loading ? "Diagnostic in progress" : ready === false ? "Service unavailable" : "Run diagnostic"}</span>
              <span aria-hidden="true">RUN ↗</span>
            </button>
            {!formValid && <p className="form-note">Correct highlighted values before running.</p>}
          </form>

          <section className={`diagnostic-panel panel state-${consoleState}`} aria-live="polite">
            <div className="panel-title">
              <div><p className="kicker"><span>02</span> Diagnostic output</p><h2>Machine condition</h2></div>
              {result && !loading && <code>{result.request_id.slice(0, 8).toUpperCase()}</code>}
            </div>

            <div className={`alarm-banner ${consoleState}`} role={error || critical ? "alert" : "status"}>
              <div className="alarm-icon" aria-hidden="true"><i /></div>
              <div><strong>{statusCopy[consoleState][0]}</strong><span>{statusCopy[consoleState][1]}</span></div>
              {ready === false && <button type="button" onClick={() => checkHealth()}>Retry connection</button>}
            </div>

            {loading ? (
              <div className="processing-state" aria-label="Diagnostic pipeline is running">
                <div className="scan-field" aria-hidden="true"><i /></div>
                <div><strong>Processing sensor vector</strong><p>Retrieval and generation start only if risk exceeds the decision threshold.</p></div>
              </div>
            ) : !result ? (
              <div className="empty-state">
                <div className="signal-grid" aria-hidden="true"><i /><i /><i /><i /><i /></div>
                <div><strong>No diagnostic sample</strong><p>The console is ready to evaluate five live sensor measurements.</p></div>
              </div>
            ) : (
              <div className="diagnostic-data">
                <div className="risk-instrument">
                  <div className="risk-readout"><span>Failure probability</span><strong>{risk.toFixed(2)}<small>%</small></strong></div>
                  <div className="risk-scale">
                    <div className="risk-labels"><span>0</span><span>Decision threshold {threshold.toFixed(2)}%</span><span>100</span></div>
                    <div className="risk-track">
                      <i className="risk-fill" style={{ width: `${Math.max(Math.min(risk, 100), 0.35)}%` }} />
                      <b className="threshold-marker" style={{ left: `${Math.min(Math.max(threshold, 0), 100)}%` }} />
                    </div>
                  </div>
                </div>

                <div className="diagnostic-grid">
                  <section className="failure-panel" aria-labelledby="failure-heading">
                    <div className="subpanel-heading"><h3 id="failure-heading">Failure signatures</h3><span>{result.primary_failure ?? "NONE ACTIVE"}</span></div>
                    <div className="failure-list">
                      {rankedFailures.map((failure) => {
                        const probability = failure.probability * 100;
                        return (
                          <div key={failure.failure_type}>
                            <span>{failure.failure_type}</span><div><i style={{ width: `${Math.max(probability, 0.25)}%` }} /></div><strong>{probability.toFixed(1)}%</strong>
                          </div>
                        );
                      })}
                    </div>
                  </section>

                  <section className="route-panel" aria-labelledby="route-heading">
                    <div className="subpanel-heading"><h3 id="route-heading">Decision route</h3><span>{critical ? "ESCALATED" : "BYPASS ACTIVE"}</span></div>
                    <ol>
                      <li className="complete"><span>01</span>ML classification<strong>Complete</strong></li>
                      <li className={critical ? "complete" : "bypassed"}><span>02</span>Evidence retrieval<strong>{critical ? "Complete" : "Bypassed"}</strong></li>
                      <li className={critical ? "complete" : "bypassed"}><span>03</span>Local generation<strong>{critical ? "Complete" : "Bypassed"}</strong></li>
                    </ol>
                  </section>
                </div>

                <div className="telemetry-strip" aria-label="Pipeline latency">
                  <div><span>ML inference</span><strong>{formatMs(result.timings.ml_ms)}</strong></div>
                  <div><span>Retrieval</span><strong>{formatMs(result.timings.retrieval_ms)}</strong></div>
                  <div><span>Generation</span><strong>{formatMs(result.timings.generation_ms)}</strong></div>
                  <div className="total"><span>Total latency</span><strong>{formatMs(result.timings.total_ms)}</strong></div>
                </div>
              </div>
            )}
          </section>
        </div>

        {result?.mitigation && !loading && (
          <section className={`mitigation-section ${result.mitigation.status === "MANUAL_REVIEW_REQUIRED" ? "lockout" : ""}`}>
            <div className="mitigation-header">
              <div><p className="kicker"><span>03</span> Safety response</p><h2>{result.mitigation.status === "MITIGATION_READY" ? "Grounded mitigation" : "Manual review required"}</h2></div>
              <div className="authorization-flag"><i aria-hidden="true" /> Human authorization required</div>
            </div>
            {result.mitigation.mitigation_plan ? (
              <div className="mitigation-grid">
                <article className="immediate-action"><span>Immediate action</span><p>{result.mitigation.mitigation_plan.immediate_action}</p></article>
                <article><span>Required PPE</span><ul>{result.mitigation.mitigation_plan.required_ppe.map((item) => <li key={item}>{item}</li>)}</ul></article>
                <article><span>Safety sequence</span><ol>{result.mitigation.mitigation_plan.safety_steps.map((item) => <li key={item}>{item}</li>)}</ol></article>
                <article>
                  <span>Verified sources</span>
                  <ul className="citations">
                    {result.mitigation.mitigation_plan.citations.map((citation) => (
                      <li key={`${citation.document_id}-${citation.page}`}><b>{citation.document_id}</b><em>PAGE {citation.page}</em><small>{citation.title}</small></li>
                    ))}
                  </ul>
                </article>
              </div>
            ) : (
              <div className="manual-review" role="alert"><strong>DO NOT ACT ON AUTOMATED OUTPUT</strong><p>{result.mitigation.reason}</p></div>
            )}
          </section>
        )}
      </main>

      <footer><span>PRISM OPS / v0.1.0</span><p>Decision support only · Verify isolation before machine intervention</p></footer>
    </>
  );
}
