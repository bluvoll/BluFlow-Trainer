import { Component, useEffect, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { MetricsView } from './Metrics';
import 'dockview-react/dist/styles/dockview.css';
import './style.css';
import type { Metrics } from './types';

function Snapshot({ metrics }: { metrics: Metrics }) {
  useEffect(() => {
    let cancelled = false;
    let frame = 0;
    let last = '';
    let stable = 0;
    // ResizeObserver schedules canvas drawing separately from React's commit.
    // Wait for every plot at its final dimensions, not just the surrounding text.
    const check = () => {
      if (cancelled) return;
      const canvases = [
        ...document.querySelectorAll<HTMLCanvasElement>('.snapshot-metrics canvas'),
      ];
      const charts = canvases.map((canvas) => {
        const { x, y, width, height } = canvas.getBoundingClientRect();
        return { x, y, width, height };
      });
      const key = JSON.stringify(charts);
      const painted =
        canvases.length === 4 &&
        canvases.every(
          (canvas, i) =>
            canvas.dataset.rendered === 'true' && charts[i].width > 0 && charts[i].height > 0,
        );
      stable = painted && key === last ? stable + 1 : 0;
      last = key;
      if (stable >= 2) window.bluflow.ready({ charts, width: innerWidth, height: innerHeight });
      else frame = requestAnimationFrame(check);
    };
    void document.fonts.ready.then(() => {
      if (!cancelled) frame = requestAnimationFrame(check);
    });
    return () => {
      cancelled = true;
      cancelAnimationFrame(frame);
    };
  }, []);
  return <MetricsView metrics={metrics} snapshot />;
}

class Boundary extends Component<{ children: ReactNode }, { error: string }> {
  state = { error: '' };
  static getDerivedStateFromError(error: Error) {
    return { error: error.message };
  }
  render() {
    return this.state.error ? (
      <div className="startup-screen">
        <h1>Workspace could not render</h1>
        <p role="alert">{this.state.error}</p>
        <p>
          Restart BluFlow-Trainer. If a saved layout caused this, remove Electron’s workspace.json.
        </p>
      </div>
    ) : (
      this.props.children
    );
  }
}
const root = createRoot(document.getElementById('root')!);
if (new URLSearchParams(location.search).has('snapshot')) {
  void window.bluflow.snapshot().then((metrics) => {
    root.render(<Snapshot metrics={metrics} />);
  });
} else
  root.render(
    <Boundary>
      <App />
    </Boundary>,
  );
