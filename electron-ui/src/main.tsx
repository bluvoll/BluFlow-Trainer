import { Component, useEffect, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { MetricsView } from './Metrics';
import 'dockview-react/dist/styles/dockview.css';
import './style.css';
import type { Metrics } from './types';

function Snapshot({ metrics }: { metrics: Metrics }) {
  useEffect(() => {
    void document.fonts.ready.then(() =>
      requestAnimationFrame(() => requestAnimationFrame(() => window.bluflow.ready())),
    );
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
