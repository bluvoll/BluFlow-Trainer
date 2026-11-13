import { useEffect, useRef, useState, type ReactNode } from 'react';
import { useActions } from './actions';
import { CloseIcon, PathInput } from './Fields';
import { command, error, subscribeEvents, useStore } from './store';

export function Dialog({
  title,
  onClose,
  children,
  footer,
  wide = false,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    const previous = document.activeElement as HTMLElement;
    ref.current?.focus();
    const key = (event: KeyboardEvent) => {
      if (Array.from(document.querySelectorAll('[role="dialog"]')).at(-1) !== ref.current) return;
      if (event.key === 'Escape') {
        event.preventDefault();
        close.current();
      }
      if (event.key === 'Tab' && ref.current) {
        const nodes = Array.from(
          ref.current.querySelectorAll<HTMLElement>(
            'button:not(:disabled),input:not(:disabled),select:not(:disabled),textarea:not(:disabled),[tabindex="0"]',
          ),
        );
        const first = nodes[0],
          last = nodes.at(-1);
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    };
    window.addEventListener('keydown', key);
    return () => {
      window.removeEventListener('keydown', key);
      previous?.focus();
    };
  }, []);
  return (
    <div className="dialog-scrim">
      <div
        ref={ref}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        className={'dialog ' + (wide ? 'dialog-wide' : '')}
      >
        <div className="window-heading">
          <strong>{title}</strong>
          <button aria-label="Close dialog" className="window-close" onClick={onClose}>
            <CloseIcon />
          </button>
        </div>
        <div className="dialog-body">{children}</div>
        {footer && <div className="dialog-footer">{footer}</div>}
      </div>
    </div>
  );
}

export function Preferences({ onClose }: { onClose: () => void }) {
  const { state } = useStore((value) => ({ state: value.state })),
    { ask, start } = useActions();
  const [folder, setFolder] = useState(state?.configDir || ''),
    [experimental, setExperimental] = useState(!!state?.preferences.show_experimental),
    [allocator, setAllocator] = useState(state?.preferences.windows_allocator || 'inherit');
  const save = () =>
    void command('preferences.set', {
      patch: { config_dir: folder, show_experimental: experimental, windows_allocator: allocator },
    })
      .then(onClose)
      .catch(error);
  return (
    <Dialog
      title="Preferences"
      onClose={onClose}
      footer={
        <>
          <button onClick={onClose}>Cancel</button>
          <button className="primary" onClick={save}>
            Save preferences
          </button>
        </>
      }
    >
      <label className="dialog-field">
        <span>TOML save folder</span>
        <PathInput value={folder} kind="folder" label="TOML save folder" onChange={setFolder} />
      </label>
      <button onClick={() => setFolder('')}>Use repository default</button>
      <p className="hint">
        Existing configurations stay in place. Opening external TOMLs and saving makes a copy in the
        managed folder.
      </p>
      <label className="check">
        <input
          type="checkbox"
          checked={experimental}
          onChange={(e) => setExperimental(e.target.checked)}
        />
        Enable advanced experimental settings
      </label>
      <p className="hint">
        Active experiments always reveal their controls. This preference only changes visibility.
      </p>
      {state?.platform === 'win32' && (
        <>
          <label className="dialog-field">
            <span>CUDA memory allocator</span>
            <select value={allocator} onChange={(e) => setAllocator(e.target.value)}>
              <option value="inherit">Default · respect environment, otherwise native</option>
              <option value="native">Native · compatibility</option>
              <option value="cudaMallocAsync">cudaMallocAsync · optional memory savings</option>
            </select>
          </label>
          <p className="hint">
            Applies to new training, caching, and calibration jobs. Explicit selections override
            allocator environment settings.
          </p>
          <button
            disabled={state.busy}
            onClick={() =>
              void ask(
                'Install English C++ language pack',
                'This modifies the Visual Studio C++ installation and may require administrator approval. Continue?',
                'Install',
              )
                .then(async (yes) => {
                  if (yes) await start('cpp-english');
                })
                .catch(error)
            }
          >
            Install English C++ language pack
          </button>
        </>
      )}
      <p className="hint">Panel arrangement and window size are saved separately for Electron.</p>
    </Dialog>
  );
}

const toolTitles: Record<string, string> = {
  alpha: 'Edit LoRA alpha / strength',
  qkv: 'Convert MOSS Q/K/V',
  resize: 'Resize LoRA rank',
  recover: 'Recover training TOML',
};
export function UtilityDialog({ tool, onClose }: { tool: string | null; onClose: () => void }) {
  const { ask } = useActions();
  const [source, setSource] = useState(''),
    [destination, setDestination] = useState(''),
    [base, setBase] = useState(''),
    [mode, setMode] = useState('bake'),
    [value, setValue] = useState(1.5),
    [layout, setLayout] = useState('fused'),
    [method, setMethod] = useState('sv_fro'),
    [threshold, setThreshold] = useState(0.99),
    [cap, setCap] = useState(false),
    [rank, setRank] = useState(8);
  const [running, setRunning] = useState(false),
    [result, setResult] = useState<any>(null),
    [failure, setFailure] = useState(''),
    [logs, setLogs] = useState<string[]>([]),
    [inspection, setInspection] = useState<any>(null);
  const token = useRef(''),
    purpose = useRef('');
  useEffect(
    () =>
      subscribeEvents((event) => {
        if (event.id !== token.current) return;
        if (event.event === 'utility_log') setLogs((lines) => [...lines, event.text].slice(-100));
        if (event.event === 'utility_result') {
          setRunning(false);
          if (event.error) setFailure(event.error);
          else if (purpose.current === 'alpha-inspect') setInspection(event.result);
          else setResult(event.result);
        }
      }),
    [],
  );
  useEffect(() => {
    if (!running) {
      setResult(null);
      setFailure('');
      setLogs([]);
    }
  }, [tool]);
  if (!tool) return null;
  const launch = async (kind: string, args: any) => {
    setRunning(true);
    setFailure('');
    setResult(null);
    setLogs([]);
    purpose.current = kind;
    try {
      let result = await command('utility.run', { kind, args });
      if (result.confirm) {
        if (!(await ask('Output already exists', result.message, 'Overwrite output'))) {
          setRunning(false);
          return;
        }
        result = await command('utility.run', { kind, args: { ...args, overwrite: true } });
      }
      token.current = result.id;
    } catch (exc) {
      setRunning(false);
      setFailure(String(exc));
    }
  };
  const run = async () => {
    let target = destination;
    if (tool === 'alpha' && !target)
      target =
        source.replace(/(\.[^./\\]+)$/, '') + `-${mode}${mode === 'dim' ? '' : value}.safetensors`;
    await launch(tool, {
      source,
      destination: target || null,
      base: base || null,
      mode,
      value,
      layout,
      method,
      threshold,
      rank: method === 'fixed' || cap ? rank : null,
    });
  };
  const close = () => {
    if (!running) onClose();
  };
  return (
    <Dialog
      title={toolTitles[tool]}
      onClose={close}
      wide
      footer={
        <>
          <span className="muted">
            {running ? 'Working in a separate CPU process…' : 'Creates a separate output file.'}
          </span>
          <button disabled={running} onClick={onClose}>
            Close
          </button>
          <button
            className="primary"
            disabled={running || !source}
            onClick={() => void run().catch(error)}
          >
            {running ? 'Working…' : tool === 'recover' ? 'Recover TOML' : 'Write output'}
          </button>
        </>
      }
    >
      <label className="dialog-field">
        <span>Source adapter / checkpoint</span>
        <PathInput
          label="Utility source"
          value={source}
          onChange={(v) => {
            setSource(v);
            setInspection(null);
          }}
        />
      </label>
      <label className="dialog-field">
        <span>Output file · blank chooses a new filename</span>
        <div className="path-input">
          <input
            aria-label="Utility destination"
            value={destination}
            onChange={(e) => setDestination(e.target.value)}
          />
          <button
            onClick={() =>
              void window.bluflow
                .file({
                  kind: tool === 'recover' ? 'save-toml' : 'save-model',
                  defaultPath: destination || undefined,
                })
                .then((path) => path && setDestination(path))
                .catch(error)
            }
          >
            Browse
          </button>
        </div>
      </label>
      {tool === 'alpha' && (
        <>
          <div className="button-row">
            <button
              disabled={!source || running}
              onClick={() => void launch('alpha-inspect', { source }).catch(error)}
            >
              Inspect module groups
            </button>
            {inspection && (
              <span>
                {inspection.modules} modules · {inspection.dora} DoRA tensors
              </span>
            )}
          </div>
          {inspection && (
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Alpha</th>
                  <th>Modules</th>
                </tr>
              </thead>
              <tbody>
                {inspection.groups.map((group: any[], i: number) => (
                  <tr key={i}>
                    {group.map((v, j) => (
                      <td key={j}>{String(v ?? 'Unknown')}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <label className="dialog-field">
            <span>Operation</span>
            <select
              value={mode}
              onChange={(e) => {
                setMode(e.target.value);
                setValue(e.target.value === 'bake' ? 1.5 : 8);
              }}
            >
              <option value="bake">Bake inference strength into alpha</option>
              <option value="set">Set alpha to a value</option>
              <option value="dim">Set alpha = rank</option>
            </select>
          </label>
          <label className="dialog-field">
            <span>{mode === 'bake' ? 'Inference strength' : 'Alpha'}</span>
            <input
              type="number"
              min="0"
              step="any"
              value={value}
              disabled={mode === 'dim'}
              onChange={(e) => setValue(Number(e.target.value))}
            />
          </label>
          <label className="dialog-field">
            <span>DoRA training base model · optional for exact bake</span>
            <PathInput label="DoRA base model" value={base} onChange={setBase} />
          </label>
          <p className="hint">
            Plain LoRA strength is alpha / rank. DoRA needs its training base model to bake
            inference strength exactly; changing alpha alone changes its normalization.
          </p>
        </>
      )}
      {tool === 'qkv' && (
        <>
          <label className="dialog-field">
            <span>Output QKV format</span>
            <select value={layout} onChange={(e) => setLayout(e.target.value)}>
              <option value="fused">Fused</option>
              <option value="separated">Separated</option>
            </select>
          </label>
          <p className="hint">
            Reorganizes MOSS Q/K/V projections while preserving the represented adapter update.
          </p>
        </>
      )}
      {tool === 'resize' && (
        <>
          <label className="dialog-field">
            <span>Resize method</span>
            <select value={method} onChange={(e) => setMethod(e.target.value)}>
              <option value="sv_fro">sv_fro · retain Frobenius norm</option>
              <option value="fixed">Fixed rank</option>
            </select>
          </label>
          <label className="dialog-field">
            <span>Frobenius norm retention</span>
            <input
              aria-label="Frobenius norm retention"
              type="number"
              min="0"
              max="1"
              step=".001"
              value={threshold}
              disabled={method !== 'sv_fro'}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={cap}
              disabled={method !== 'sv_fro'}
              onChange={(e) => setCap(e.target.checked)}
            />
            Cap the selected rank
          </label>
          <label className="dialog-field">
            <span>{method === 'fixed' ? 'Target rank' : 'Maximum rank'}</span>
            <input
              type="number"
              min="1"
              value={rank}
              disabled={method !== 'fixed' && !cap}
              onChange={(e) => setRank(Number(e.target.value))}
            />
          </label>
          <p className="hint">
            0.99 retains at least 98.01% squared energy before output rounding. A rank cap can miss
            that target; the report identifies affected modules. Existing ranks never increase.
          </p>
        </>
      )}
      {tool === 'recover' && (
        <p className="hint">
          Reads embedded BluFlow training metadata without loading weights. Restores the recipe and
          clears resume instructions. Review local paths and output names before training.
        </p>
      )}
      {failure && <pre className="tool-error">{failure}</pre>}
      {result && (
        <div className="tool-result">
          <strong>Output written</strong>
          <p className="path-text">{result.path}</p>
          {result.notes?.map((note: string, i: number) => (
            <p key={i}>{note}</p>
          ))}
          {result.report && (
            <pre>
              {typeof result.report === 'string'
                ? result.report
                : JSON.stringify(result.report, null, 2)}
            </pre>
          )}
        </div>
      )}
      {logs.length > 0 && <pre className="utility-log">{logs.join('\n')}</pre>}
    </Dialog>
  );
}
