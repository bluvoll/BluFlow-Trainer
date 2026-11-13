import { useEffect, useState } from 'react';
import Chart from './Chart';
import { command, error, setPaused, useStore } from './store';
import type { Flat, Metrics } from './types';

export function elapsedText(milliseconds: number | null) {
  if (milliseconds === null) return '—';
  const seconds = Math.max(0, Math.floor(milliseconds / 1000));
  return [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60]
    .map((v) => String(v).padStart(2, '0'))
    .join(':');
}
type Series = [string, (record: Flat) => number | undefined][];
const LOSS: Series = [
  ['Loss', (r) => r.loss],
  ['Loss EMA', (r) => r.ema],
  ['MSE', (r) => r.mse],
  ['HF', (r) => r.hf],
];
const LR: Series = [['LR', (r) => r.lr]],
  SPEED: Series = [['s/it', (r) => r.sit]];
const SINGLE_MEMORY: Series = [['rank 0', (r) => r.peak?.[0]]],
  MULTI_MEMORY: Series = [...SINGLE_MEMORY, ['rank 1', (r) => r.peak?.[1]]];
export function MetricsView({
  metrics,
  snapshot = false,
  paused = false,
}: {
  metrics: Metrics;
  snapshot?: boolean;
  paused?: boolean;
}) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  const latest = metrics.latest,
    elapsed = metrics.startedAt ? now - metrics.startedAt : metrics.elapsed;
  const memory = metrics.records.some((r) => r.peak?.length > 1) ? MULTI_MEMORY : SINGLE_MEMORY;
  return (
    <div className={'metrics-view ' + (snapshot ? 'snapshot-metrics' : '')} data-testid="metrics">
      <div className="metrics-top">
        <div>
          <span className="eyebrow">{snapshot ? 'FINAL TRAINING METRICS' : 'LIVE METRICS'}</span>
          {latest.step !== undefined && (
            <h2>
              Step {latest.step} / {latest.total}
            </h2>
          )}
          <div className="metrics-facts">
            <span>
              Elapsed <b data-testid="elapsed">{elapsedText(elapsed)}</b>
            </span>
            {latest.loss !== undefined && (
              <>
                <span>
                  Loss{' '}
                  <b>
                    {typeof latest.loss === 'number'
                      ? latest.loss.toPrecision(5)
                      : String(latest.loss)}
                  </b>
                </span>
                <span>
                  Step <b>{latest.sit}s</b>
                </span>
                {latest.eta && (
                  <span>
                    ETA <b>{latest.eta}</b>
                  </span>
                )}
                {latest.epoch !== undefined && (
                  <span>
                    Epoch <b>{latest.epoch}</b>
                  </span>
                )}
                {latest.lr !== undefined && (
                  <span>
                    LR{' '}
                    <b>
                      {(latest.lrs || [latest.lr])
                        .map((lr: number) => lr.toExponential(2))
                        .join(' / ')}
                    </b>
                  </span>
                )}
                {latest.hf !== undefined && (
                  <span>
                    HF <b>{latest.hf.toPrecision(4)}</b>
                  </span>
                )}
                {latest.ot !== undefined && (
                  <span>
                    OT <b>{latest.ot.toFixed(2)}</b>
                  </span>
                )}
                {latest.peak?.length > 0 && (
                  <span>
                    Peak allocated{' '}
                    <b>{latest.peak.map((value: number) => value.toFixed(2)).join(' / ')} GiB</b>
                  </span>
                )}
              </>
            )}
            {latest.resolution_stage && (
              <span>
                Stage {latest.resolution_stage.stage ?? latest.resolution_stage.index ?? ''} ·{' '}
                {latest.resolution_stage.resolution}px
                {' · '}
                {latest.resolution_stage.completed}/{latest.resolution_stage.total} updates
                {' · '}
                {latest.resolution_stage.images} images
                {' · '}effective batch {latest.resolution_stage.effective_batch}
              </span>
            )}
          </div>
          <div className="metrics-facts memory-facts">
            {[
              ['allocated', 'Allocated'],
              ['reserved', 'Reserved'],
              ['device_used', 'Device used · all processes'],
            ].map(
              ([key, label]) =>
                latest[key] && (
                  <span key={key}>
                    {label}{' '}
                    <b>{latest[key].map((value: number) => value.toFixed(2)).join(' / ')} GiB</b>
                  </span>
                ),
            )}
          </div>
        </div>
        {!snapshot && (
          <div className="button-row">
            <button
              onClick={() =>
                void command('metrics.control', { action: 'pause' })
                  .then((r) => setPaused(r.paused))
                  .catch(error)
              }
            >
              {paused ? 'Resume' : 'Pause'}
            </button>
            <button
              onClick={() => void command('metrics.control', { action: 'clear' }).catch(error)}
            >
              Clear
            </button>
          </div>
        )}
      </div>
      <div className="chart-grid">
        <Chart title="Loss" records={metrics.records} series={LOSS} />
        <Chart
          title="Learning rate"
          records={metrics.records}
          series={LR}
          format={(n) => n.toExponential(1)}
        />
        <Chart title="Step time · s/it" records={metrics.records} series={SPEED} />
        <Chart title="Peak allocated VRAM · GiB" records={metrics.records} series={memory} />
      </div>
    </div>
  );
}
export default function MetricsPanel() {
  const { metrics, paused } = useStore((value) => ({
    metrics: value.metrics,
    paused: value.paused,
  }));
  return <MetricsView metrics={metrics} paused={paused} />;
}
