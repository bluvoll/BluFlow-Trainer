import { useEffect, useRef, useState } from 'react';
import { useActions } from './actions';
import { chooseCategory, command, edit, error, subscribeEvents, useStore } from './store';
import FieldRow from './Fields';
import MetricsPanel from './Metrics';

export function Navigation() {
  const { data, state, category } = useStore((value) => ({
      data: value.data,
      state: value.state,
      category: value.category,
    })),
    { search, setSearch, reveal } = useActions();
  if (!data || !state) return null;
  return (
    <div className="navigation-panel">
      <div className="panel-intro">
        <span className="eyebrow">CONFIGURATION</span>
        <h2>Training recipe</h2>
        <span className="muted">{Object.keys(data.fields).length} settings · one workspace</span>
      </div>
      <input
        className="search-input"
        type="search"
        aria-label="Search settings"
        placeholder="Find a setting…"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />
      <div className="section-navigation">
        {data.pages
          .filter(([title]) => title !== 'Experimental' || state.validation.experimental)
          .map(([title, groups]) => (
            <button
              key={title}
              className={category === title ? 'selected' : ''}
              onClick={() => {
                chooseCategory(title);
                reveal('settings');
              }}
            >
              <span className="navigation-icon">
                {title === 'Dataset' ? (
                  <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor">
                    <rect x="3" y="3" width="14" height="14" rx="2" />
                    <path d="m4 14 4-4 3 3 2-2 4 4" />
                    <circle cx="13" cy="7" r="1" />
                  </svg>
                ) : title === 'Training' ? (
                  <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor">
                    <path d="m6 3 10 7-10 7z" />
                  </svg>
                ) : (
                  <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor">
                    <path d="M3 5h14M3 10h14M3 15h14" />
                    <circle cx="7" cy="5" r="2" />
                    <circle cx="13" cy="10" r="2" />
                    <circle cx="8" cy="15" r="2" />
                  </svg>
                )}
              </span>
              <span>
                {title}
                <small>{groups.length} sections</small>
              </span>
              <span className="nav-count">
                {groups.reduce((sum, [, keys]) => sum + keys.length, 0)}
              </span>
            </button>
          ))}
      </div>
      <div className="navigation-note">
        <b>Portable configurations</b>
        <p>Your training settings use the same TOML format as the original trainer.</p>
        <span className="path-text">{state.configDir}</span>
      </div>
    </div>
  );
}

export function Settings() {
  const { data, state, category } = useStore((value) => ({
      data: value.data,
      state: value.state,
      category: value.category,
    })),
    { search, setSearch, start } = useActions();
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  useEffect(() => {
    void window.bluflow
      .preferences()
      .then((p) => setCollapsed(new Set(p.collapsed || [])))
      .catch(error);
  }, []);
  if (!data || !state) return null;
  const pages = search.trim()
    ? data.pages.filter(([title]) => title !== 'Experimental' || state.validation.experimental)
    : data.pages.filter(
        ([title]) =>
          title === category && (title !== 'Experimental' || state.validation.experimental),
      );
  const matching = pages
    .flatMap(([page, groups]) =>
      groups.map(([title, keys]) => ({
        page,
        title,
        keys: keys.filter(
          (key) =>
            !search.trim() ||
            `${data.fields[key].label} ${data.fields[key].tooltip} ${key}`
              .toLowerCase()
              .includes(search.toLowerCase()),
        ),
      })),
    )
    .filter((g) => g.keys.length);
  const toggle = (name: string) => {
    const next = new Set(collapsed);
    next.has(name) ? next.delete(name) : next.add(name);
    setCollapsed(next);
    void window.bluflow.preferences({ collapsed: [...next] }).catch(error);
  };
  const all = (collapse: boolean) => {
    const next = new Set(collapsed);
    for (const group of matching) collapse ? next.add(group.title) : next.delete(group.title);
    setCollapsed(next);
    void window.bluflow.preferences({ collapsed: [...next] }).catch(error);
  };
  return (
    <div className="settings-panel">
      <div className="content-heading">
        <div>
          <span className="eyebrow">{search ? 'SEARCH RESULTS' : 'TRAINING SETTINGS'}</span>
          <h2>{search ? `“${search}”` : category}</h2>
        </div>
        <div className="button-row">
          {search && <button onClick={() => setSearch('')}>Clear search</button>}
          <button onClick={() => all(false)}>Expand all</button>
          <button onClick={() => all(true)}>Collapse all</button>
        </div>
      </div>
      <div className="settings-groups">
        {matching.map((group) => (
          <section className="settings-group" key={group.title}>
            <button
              className="group-heading"
              aria-expanded={search ? true : !collapsed.has(group.title)}
              onClick={() => toggle(group.title)}
            >
              <span>
                {search && <small>{group.page} / </small>}
                {group.title}
              </span>
              <svg
                width="12"
                height="12"
                viewBox="0 0 12 12"
                style={{
                  transform: collapsed.has(group.title) && !search ? 'rotate(-90deg)' : undefined,
                }}
                fill="none"
                stroke="currentColor"
              >
                <path d="m3 4 3 3 3-3" />
              </svg>
            </button>
            {(!collapsed.has(group.title) || search) && (
              <div className="group-content">
                {group.keys.map((key) => (
                  <div key={key}>
                    <FieldRow field={data.fields[key]} />
                    {group.title === 'Flow Matching' && key === 'flow.use_ot' && <Distribution />}
                  </div>
                ))}
                {group.keys.includes('quant.matmul_tuning') && (
                  <button
                    disabled={
                      state.busy ||
                      !state.validation.valid ||
                      !state.validation.enabled['quant.matmul_tuning']
                    }
                    onClick={() => void start('calibrate').catch(error)}
                  >
                    Calibrate INT8 kernels
                  </button>
                )}
              </div>
            )}
          </section>
        ))}
      </div>
      {!matching.length && <div className="empty-panel">No matching settings.</div>}
    </div>
  );
}

function Distribution() {
  const { state } = useStore((value) => ({ state: value.state })),
    d = state?.validation.distribution;
  const [grid, setGrid] = useState<[number, number]>(() => state?.previewGrid || [64, 64]);
  useEffect(() => {
    const timer = setTimeout(
      () => void command('preview.timestep', { height: grid[0], width: grid[1] }).catch(error),
      180,
    );
    return () => clearTimeout(timer);
  }, [grid[0], grid[1]]);
  if (!d)
    return (
      <div className="distribution muted">Fix the timestep settings to update the preview.</div>
    );
  const max = Math.max(...d.probabilities, 1e-8);
  return (
    <div className="distribution" aria-label="Timestep distribution">
      {state?.flat['flow.flux_shift'] && (
        <div className="preview-grid" title="Preview only. Does not change training resolution.">
          <span>Preview latent H × W</span>
          {grid.map((value, index) => (
            <input
              key={index}
              aria-label={index === 0 ? 'Preview latent height' : 'Preview latent width'}
              type="number"
              min="1"
              max="512"
              value={value}
              onChange={(event) =>
                setGrid(
                  (previous) =>
                    previous.map((v, i) =>
                      i === index ? Math.max(1, Math.min(512, Number(event.target.value) || 1)) : v,
                    ) as [number, number],
                )
              }
            />
          ))}
        </div>
      )}
      <div className="distribution-bars">
        {d.probabilities.map((p, i) => (
          <div
            key={i}
            title={`${(i / d.probabilities.length).toFixed(2)}–${((i + 1) / d.probabilities.length).toFixed(2)}: ${(p * 100).toFixed(2)}%`}
            style={{ height: (p / max) * 100 + '%' }}
          />
        ))}
      </div>
      <div className="distribution-labels">
        <span>Clean · 0</span>
        <span>Median {d.median.toFixed(3)}</span>
        <span>Noise · 1</span>
      </div>
      <small>
        {(d.high_noise * 100).toFixed(1)}% at high noise · preview before curriculum mapping
      </small>
    </div>
  );
}

export function RunControls() {
  const { state } = useStore((value) => ({ state: value.state })),
    { start, addCurrent, openPreferences } = useActions();
  if (!state) return null;
  const v = state.validation,
    blocked = !v.valid || v.advisories.some(([level]) => level === 'error');
  return (
    <div className="run-panel">
      <div className="panel-intro">
        <span className="eyebrow">RUN CONTROL</span>
        <h2>{state.busy ? 'Job in progress' : 'Ready when you are'}</h2>
        <span className="muted">{state.active?.label || 'Validate, prepare, train.'}</span>
      </div>
      <section className="run-section">
        <div className="section-caption">COMPUTE</div>
        <div className="gpu-list">
          {state.gpus.map((g) => (
            <label
              key={g.index}
              title={g.name}
              className={'gpu-option ' + (state.selectedGpus.includes(g.index) ? 'chosen' : '')}
            >
              <input
                type={state.multiGpu ? 'checkbox' : 'radio'}
                name="gpu"
                checked={state.selectedGpus.includes(g.index)}
                disabled={state.busy || state.gpus.length === 1}
                onChange={(e) =>
                  void command('gpus.set', {
                    selected: state.multiGpu
                      ? e.target.checked
                        ? [...state.selectedGpus, g.index]
                        : state.selectedGpus.filter((i) => i !== g.index)
                      : [g.index],
                  }).catch(error)
                }
              />
              <span>
                GPU {g.index}
                <small>{g.name}</small>
              </span>
            </label>
          ))}
        </div>
        <p className="hint">
          {state.selectedGpus.length} process{state.selectedGpus.length === 1 ? '' : 'es'}
          {!state.multiGpu ? ' · single GPU on Windows' : ''}
        </p>
        <label className="check">
          <input
            type="checkbox"
            checked={state.preferences.autosave !== false}
            onChange={(e) =>
              void command('preferences.set', { patch: { autosave: e.target.checked } }).catch(
                error,
              )
            }
          />
          Save settings when starting
        </label>
      </section>
      <section className="run-section">
        <div className="section-caption">PREPARATION</div>
        <div className="prep-buttons">
          <button disabled={state.busy} onClick={() => void start('audit').catch(error)}>
            Audit dataset
          </button>
          <button disabled={state.busy} onClick={() => void start('cache').catch(error)}>
            Cache latents
          </button>
          <button disabled={state.busy} onClick={() => void start('dry-run').catch(error)}>
            Cache · dry run
          </button>
        </div>
      </section>
      <section className="run-section validation">
        <div className="section-caption">CONFIGURATION STATUS</div>
        <div className={v.valid ? 'validation-ok' : 'validation-error'}>
          {v.valid ? 'Configuration valid' : v.error}
        </div>
        <p>{v.summary}</p>
        {v.advisories.map(([level, text], i) => (
          <p className={'advisory ' + level} key={i}>
            {text}
          </p>
        ))}
        {v.windowsRepair && (
          <button
            onClick={() => {
              edit('train.attention_backend', 'sdpa');
              edit('train.pack_resolutions', false);
            }}
          >
            Use SDPA and disable native-resolution packing
          </button>
        )}
      </section>
      <div className="run-actions">
        {state.busy ? (
          <button className="danger primary" onClick={() => void command('job.stop').catch(error)}>
            Stop
          </button>
        ) : (
          <button
            className="primary"
            disabled={blocked}
            onClick={() => void start('train').catch(error)}
          >
            Start Training
          </button>
        )}
        <button disabled={blocked} onClick={() => void addCurrent().catch(error)}>
          Save &amp; add to queue
        </button>
        <button className="text-button" onClick={openPreferences}>
          Preferences
        </button>
      </div>
    </div>
  );
}

export function Queue() {
  const { state } = useStore((value) => ({ state: value.state })),
    { ask } = useActions();
  const [selected, setSelected] = useState<number[]>([]),
    [continuing, setContinuing] = useState(false);
  if (!state) return null;
  const chosen = state.queue.find((e) => selected.includes(e.id));
  const perform = (operation: string, direction = 0) =>
    void command('queue.edit', { operation, ids: selected, direction }).catch(error);
  return (
    <div className="queue-panel">
      <div className="queue-toolbar">
        <div className="button-row">
          <button
            onClick={() =>
              void window.bluflow
                .file({ kind: 'toml', multiple: true, defaultPath: state.configDir })
                .then((paths) => paths && command('queue.add', { paths }))
                .catch(error)
            }
          >
            Add TOMLs
          </button>
          <button disabled={!selected.length} onClick={() => perform('move', -1)}>
            Move up
          </button>
          <button disabled={!selected.length} onClick={() => perform('move', 1)}>
            Move down
          </button>
          <button disabled={!selected.length} onClick={() => perform('remove')}>
            Remove selected
          </button>
          <button onClick={() => perform('clear')}>Clear pending</button>
          <button disabled={!selected.length} onClick={() => perform('retry')}>
            Retry selected
          </button>
        </div>
        <div className="button-row">
          <label className="check">
            <input
              type="checkbox"
              checked={continuing}
              disabled={state.busy}
              onChange={(e) => setContinuing(e.target.checked)}
            />
            Continue after errors
          </label>
          <button
            className="primary"
            disabled={state.busy || !state.queue.some((e) => e.status === 'Pending')}
            onClick={() => void command('queue.start', { continueErrors: continuing }).catch(error)}
          >
            Start queue
          </button>
        </div>
      </div>
      <div className="queue-summary">
        {['Pending', 'Caching', 'Training', 'Completed', 'Failed', 'Stopped', 'Skipped'].map(
          (status) => (
            <span key={status}>
              {status} <b>{state.queue.filter((e) => e.status === status).length}</b>
            </span>
          ),
        )}
      </div>
      <div className="queue-list">
        {state.queue.length ? (
          <table>
            <thead>
              <tr>
                <th />
                <th>Status</th>
                <th>Source configuration</th>
              </tr>
            </thead>
            <tbody>
              {state.queue.map((entry) => (
                <tr
                  key={entry.id}
                  className={selected.includes(entry.id) ? 'selected' : ''}
                  onClick={() => setSelected([entry.id])}
                >
                  <td>
                    <input
                      aria-label={'Select ' + entry.source}
                      type="checkbox"
                      checked={selected.includes(entry.id)}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) =>
                        setSelected(
                          e.target.checked
                            ? [...selected, entry.id]
                            : selected.filter((id) => id !== entry.id),
                        )
                      }
                    />
                  </td>
                  <td>
                    <span className={'status-badge ' + entry.status.toLowerCase()}>
                      {entry.status}
                    </span>
                  </td>
                  <td title={entry.source}>{entry.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="empty-panel">
            <strong>No runs queued</strong>
            <span>Add TOMLs or save the current settings to create independent snapshots.</span>
          </div>
        )}
      </div>
      {chosen && (
        <div className="queue-details">
          <span>Source: {chosen.source}</span>
          <span>Snapshot: {chosen.snapshot}</span>
          {chosen.error && <pre className="validation-error">{chosen.error}</pre>}
        </div>
      )}
      <div className="panel-footnote">
        Queued settings are session snapshots. Later source edits do not change them. Stop cancels
        the remaining jobs.
      </div>
    </div>
  );
}

export function Console() {
  const { logs, consoleTotal } = useStore((value) => ({
    logs: value.logs,
    consoleTotal: value.consoleTotal,
  }));
  const [follow, setFollow] = useState(true),
    [history, setHistory] = useState<{ start: number; lines: string[] } | null>(null),
    ref = useRef<HTMLDivElement>(null),
    historyRequest = useRef(0);
  useEffect(
    () =>
      subscribeEvents((event) => {
        if (event.event === 'console_clear') {
          historyRequest.current++;
          setHistory(null);
          setFollow(true);
        }
      }),
    [],
  );
  useEffect(() => {
    if (follow && ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [logs, follow]);
  const read = async (start?: number) => {
    const request = ++historyRequest.current;
    setFollow(false);
    const page = await command('console.read', { start, count: 900 });
    if (request !== historyRequest.current) return;
    setHistory(page);
    if (ref.current) ref.current.scrollTop = 0;
  };
  // The complete compressed history stays in Python; only one page reaches the DOM.
  const visible = history
    ? history.lines.map((text) => ({ text, progress: false }))
    : logs.slice(-900);
  const start = history?.start ?? Math.max(0, consoleTotal - visible.length);
  return (
    <div className="console-panel">
      <div className="console-toolbar">
        <span>{consoleTotal.toLocaleString()} lines</span>
        <button
          disabled={!consoleTotal || start === 0}
          onClick={() => void read(Math.max(0, start - 900)).catch(error)}
        >
          Older
        </button>
        <button
          disabled={!history || history.start + 900 >= consoleTotal}
          onClick={() => void read(start + 900).catch(error)}
        >
          Newer
        </button>
        {history && (
          <input
            className="console-position"
            type="range"
            aria-label="Console history position"
            min={0}
            max={Math.max(0, consoleTotal - 900)}
            value={start}
            onChange={(event) => void read(Number(event.target.value)).catch(error)}
          />
        )}
        <label className="check">
          <input
            type="checkbox"
            checked={follow}
            onChange={(e) => {
              if (e.target.checked) {
                historyRequest.current++;
                setHistory(null);
                setFollow(true);
              } else void read().catch(error);
            }}
          />
          Follow output
        </label>
        <button
          onClick={() =>
            void command('console.text')
              .then((text) => navigator.clipboard.writeText(text))
              .catch(error)
          }
        >
          Copy log
        </button>
        <button onClick={() => void command('console.clear').catch(error)}>Clear console</button>
      </div>
      <div className="console-output" ref={ref} role="log" aria-label="Process console">
        {visible.length ? (
          visible.map((line, i) => (
            <div className={line.progress ? 'progress-line' : ''} key={start + i}>
              {line.text}
            </div>
          ))
        ) : (
          <span className="muted">Process output appears here.</span>
        )}
      </div>
    </div>
  );
}

export const panels: Record<string, { title: string; component: () => React.JSX.Element | null }> =
  {
    navigation: { title: 'Configuration', component: Navigation },
    settings: { title: 'Settings', component: Settings },
    run: { title: 'Run controls', component: RunControls },
    metrics: { title: 'Live Metrics', component: MetricsPanel },
    queue: { title: 'Run Queue', component: Queue },
    console: { title: 'Console', component: Console },
  };
