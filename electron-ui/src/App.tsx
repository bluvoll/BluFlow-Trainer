import { useCallback, useEffect, useRef, useState } from 'react';
import {
  DockviewReact,
  type DockviewApi,
  type DockviewReadyEvent,
  type IDockviewPanelProps,
} from 'dockview-react';
import { ActionsContext } from './actions';
import { panels } from './Panels';
import { Dialog, Preferences, UtilityDialog } from './Dialogs';
import {
  clearError,
  chooseCategory,
  choosePanel,
  command,
  error,
  flush,
  initialize,
  state as store,
  subscribeEvents,
  useStore,
} from './store';

const Panel = ({ params }: IDockviewPanelProps) => {
  const Component = panels[params.kind]?.component;
  return Component ? (
    <div className="dock-panel-host">
      <Component />
    </div>
  ) : null;
};
const components = { panel: Panel };
type Question = {
  title: string;
  message: string;
  confirm: string;
  resolve: (yes: boolean) => void;
};

export default function App() {
  const {
    data,
    state,
    error: failure,
    category,
    panel,
  } = useStore((value) => ({
    data: value.data,
    state: value.state,
    error: value.error,
    category: value.category,
    panel: value.panel,
  }));
  const api = useRef<DockviewApi | null>(null),
    layoutTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const [search, setSearch] = useState(''),
    [preferences, setPreferences] = useState(false),
    [tool, setTool] = useState<string | null>(null),
    [question, setQuestion] = useState<Question | null>(null),
    [working, setWorking] = useState('');
  const ask = useCallback(
    (title: string, message: string, confirm = 'Continue') =>
      new Promise<boolean>((resolve) => setQuestion({ title, message, confirm, resolve })),
    [],
  );
  const reveal = useCallback((id: string) => {
    const dock = api.current;
    if (!dock || !panels[id]) return;
    let item = dock.getPanel(id);
    if (!item) {
      const reference = dock.getPanel(id === 'queue' ? 'console' : 'settings') || dock.panels[0];
      const direction =
        id === 'navigation'
          ? 'left'
          : id === 'run'
            ? 'right'
            : id === 'console' || (id === 'queue' && !dock.getPanel('console'))
              ? 'below'
              : 'within';
      item = dock.addPanel({
        id,
        title: panels[id].title,
        component: 'panel',
        params: { kind: id },
        renderer: 'always',
        position: reference ? { referencePanel: reference, direction } : undefined,
      });
    }
    item.api.setActive();
  }, []);
  const resetLayout = useCallback((dock: DockviewApi) => {
    dock.clear();
    dock.addPanel({
      id: 'settings',
      title: 'Settings',
      component: 'panel',
      params: { kind: 'settings' },
      renderer: 'always',
      minimumWidth: 380,
      minimumHeight: 240,
    });
    dock.addPanel({
      id: 'navigation',
      title: 'Configuration',
      component: 'panel',
      params: { kind: 'navigation' },
      renderer: 'always',
      initialWidth: 230,
      minimumWidth: 180,
      position: { referencePanel: 'settings', direction: 'left' },
    });
    dock.addPanel({
      id: 'run',
      title: 'Run controls',
      component: 'panel',
      params: { kind: 'run' },
      renderer: 'always',
      initialWidth: 285,
      minimumWidth: 230,
      position: { referencePanel: 'settings', direction: 'right' },
    });
    dock.addPanel({
      id: 'console',
      title: 'Console',
      component: 'panel',
      params: { kind: 'console' },
      renderer: 'always',
      initialHeight: 220,
      minimumHeight: 130,
      position: { referencePanel: 'settings', direction: 'below' },
    });
    dock.addPanel({
      id: 'queue',
      title: 'Run Queue',
      component: 'panel',
      params: { kind: 'queue' },
      renderer: 'always',
      position: { referencePanel: 'console', direction: 'within' },
    });
    dock.addPanel({
      id: 'metrics',
      title: 'Live Metrics',
      component: 'panel',
      params: { kind: 'metrics' },
      renderer: 'always',
      position: { referencePanel: 'settings', direction: 'within' },
    });
    dock.getPanel('settings')!.api.setActive();
    dock.getPanel('console')!.api.setActive();
  }, []);
  const ready = useCallback(
    (event: DockviewReadyEvent) => {
      const dock = event.api;
      api.current = dock;
      void window.bluflow
        .preferences()
        .then((p) => {
          if (
            p.layout &&
            Object.keys(p.layout.panels || {}).every((id) => id in panels) &&
            !p.layout.popoutGroups?.length
          ) {
            try {
              dock.fromJSON(p.layout);
            } catch {
              resetLayout(dock);
            }
          } else resetLayout(dock);
          if (p.category && data?.pages.some(([name]) => name === p.category))
            chooseCategory(p.category);
          dock.onDidLayoutChange(() => {
            clearTimeout(layoutTimer.current);
            layoutTimer.current = setTimeout(
              () => void window.bluflow.preferences({ layout: dock.toJSON() }).catch(error),
              400,
            );
          });
        })
        .catch(error);
    },
    [resetLayout, data],
  );
  const save = useCallback(
    async (destination?: string) => {
      let result = await command('config.save', { destination });
      if (result.confirm && (await ask('Config already exists', result.message, 'Overwrite')))
        result = await command('config.save', { destination: result.path, overwrite: true });
      return !result.confirm;
    },
    [ask],
  );
  const start = useCallback(
    async (action = 'train') => {
      if (action === 'cpp-english') {
        await command('job.start', { action, confirmed: true });
        return;
      }
      const options = { action, confirmed: false, overwrite: false };
      while (true) {
        const result = await command('job.start', options);
        if (!result.confirm) return;
        if (result.confirm === 'queue') {
          if (!(await ask('TOMLs are in the queue', result.message, 'Start loaded settings'))) {
            choosePanel('queue');
            reveal('queue');
            return;
          }
          options.confirmed = true;
        } else {
          if (!(await ask('Config already exists', result.message, 'Overwrite'))) return;
          options.overwrite = true;
        }
      }
    },
    [ask, reveal],
  );
  const addCurrent = useCallback(async () => {
    await flush();
    const current = await command('state');
    if (!current.validation.valid) throw Error(current.validation.error);
    const blocking = current.validation.advisories.filter(
      ([level]: [string, string]) => level === 'error',
    );
    if (blocking.length) throw Error(blocking.map(([, text]: [string, string]) => text).join('\n'));
    if (await save()) {
      const path = store().state?.path;
      if (path) {
        await command('queue.add', { paths: [path] });
        choosePanel('queue');
        reveal('queue');
      }
    }
  }, [save, reveal]);
  useEffect(() => {
    let alive = true;
    void initialize()
      .then(async () => {
        window.bluflow.ready();
        if (new URLSearchParams(location.search).has('test')) return;
        const checks = await window.bluflow.request('startup.check');
        if (!alive) return;
        const update = checks.update;
        if (update?.behind) {
          if (update.ahead || update.dirty)
            error(
              `${update.upstream} has ${update.behind} new commits. Local commits or tracked edits need manual review before updating.`,
            );
          else if (
            await ask(
              'Trainer update available',
              `${update.branch} is ${update.behind} commits behind ${update.upstream}. Update with git pull --ff-only and refresh this UI before restarting?`,
              'Update and restart',
            )
          ) {
            setWorking('Updating trainer…');
            await window.bluflow.request('startup.update', { update });
            await window.bluflow.restart({ refreshDependencies: true });
            return;
          }
        }
        if (
          checks.refreshDependencies &&
          (await ask(
            'Refresh trainer packages',
            'Installed packages differ from the repository pins, dependency files changed, or this environment has not recorded a successful refresh. Install the pinned packages and restart?',
            'Refresh and restart',
          ))
        ) {
          setWorking('Restarting to refresh packages…');
          await window.bluflow.restart({ refreshDependencies: true });
        }
      })
      .catch((exc) => {
        setWorking('');
        error(exc);
      });
    return () => {
      alive = false;
    };
  }, [ask]);
  useEffect(() => {
    if (api.current) reveal(panel);
  }, [panel, reveal]);
  useEffect(() => {
    if (api.current) reveal('settings');
  }, [category, reveal]);
  useEffect(() => {
    if (data) void window.bluflow.preferences({ category }).catch(error);
  }, [category, !!data]);
  useEffect(
    () =>
      subscribeEvents((event) => {
        if (event.event === 'activate') reveal(event.panel);
      }),
    [reveal],
  );
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
        event.preventDefault();
        void save().catch(error);
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'o') {
        event.preventDefault();
        void open();
      }
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, [save]);
  async function open() {
    try {
      const path = await window.bluflow.file({
        kind: 'toml',
        defaultPath: store().state?.configDir,
      });
      if (path) await command('config.open', { path });
    } catch (exc) {
      error(exc);
    }
  }
  async function saveAs() {
    try {
      const path = await window.bluflow.file({
        kind: 'save-toml',
        defaultPath:
          state?.path || `${state?.configDir}/${state?.flat['train.run_name'] || 'recipe'}.toml`,
      });
      if (path) await save(/\.toml$/i.test(path) ? path : path + '.toml');
    } catch (exc) {
      error(exc);
    }
  }
  const answer = (yes: boolean) => {
    question?.resolve(yes);
    setQuestion(null);
  };
  if (!data || !state)
    return (
      <div className="startup-screen">
        <div className="brand-emblem">B</div>
        <h1>BluFlow-Trainer</h1>
        <p>{failure || 'Loading your training workspace…'}</p>
      </div>
    );
  return (
    <ActionsContext.Provider
      value={{
        search,
        setSearch,
        ask,
        openPreferences: () => setPreferences(true),
        openUtility: setTool,
        reveal,
        save: async () => {
          await save();
        },
        start,
        addCurrent,
      }}
    >
      <div className="app" data-testid="app">
        <header className="app-toolbar">
          <div className="config-actions">
            <select
              aria-label="Configuration preset"
              value={state.path || ''}
              onChange={(e) => void command('config.open', { path: e.target.value }).catch(error)}
            >
              <option value="" disabled>
                Unsaved recipe
              </option>
              {state.presets.map((p) => (
                <option value={p.path} key={p.path}>
                  {p.name}
                </option>
              ))}
            </select>
            <button onClick={() => void open()}>Open</button>
            <button onClick={() => void save().catch(error)}>Save</button>
            <button onClick={() => void saveAs()}>Save As</button>
            <button
              disabled={!state.path}
              onClick={() => void command('config.open', { path: state.path }).catch(error)}
            >
              Reload
            </button>
          </div>
          <div className="toolbar-menus">
            <details className="toolbar-menu">
              <summary>Panels</summary>
              <div className="popover">
                {Object.entries(panels).map(([id, entry]) => (
                  <div className="panel-menu-row" key={id}>
                    <button
                      onClick={(e) => {
                        reveal(id);
                        e.currentTarget.closest('details')?.removeAttribute('open');
                      }}
                    >
                      {entry.title}
                    </button>
                    <button
                      title={'Float ' + entry.title}
                      onClick={() => {
                        reveal(id);
                        const p = api.current?.getPanel(id);
                        if (p)
                          api.current?.addFloatingGroup(p, {
                            width: id === 'navigation' ? 260 : 650,
                            height: 450,
                          });
                      }}
                    >
                      Float
                    </button>
                    <button
                      title={'Hide ' + entry.title}
                      onClick={() => {
                        const p = api.current?.getPanel(id);
                        if (p) api.current?.removePanel(p);
                      }}
                    >
                      Hide
                    </button>
                  </div>
                ))}
                <button
                  className="menu-divider"
                  onClick={(e) => {
                    if (api.current) resetLayout(api.current);
                    e.currentTarget.closest('details')?.removeAttribute('open');
                  }}
                >
                  Reset workspace layout
                </button>
              </div>
            </details>
            <details className="toolbar-menu">
              <summary>Utilities</summary>
              <div className="popover">
                {[
                  ['alpha', 'Edit LoRA alpha / strength'],
                  ['qkv', 'Convert MOSS Q/K/V'],
                  ['resize', 'Resize LoRA rank (SVD)'],
                  ['recover', 'Recover training TOML'],
                ].map(([kind, title]) => (
                  <button
                    key={kind}
                    onClick={(e) => {
                      setTool(kind);
                      e.currentTarget.closest('details')?.removeAttribute('open');
                    }}
                  >
                    {title}
                  </button>
                ))}
              </div>
            </details>
            <button onClick={() => setPreferences(true)}>Preferences</button>
          </div>
        </header>
        {failure && (
          <div className="error-banner" role="alert">
            <span>{failure}</span>
            <button onClick={clearError} aria-label="Dismiss message">
              Dismiss
            </button>
          </div>
        )}
        <main className="workspace">
          <DockviewReact
            className="dockview-theme-bluflow"
            components={components}
            onReady={ready}
            floatingGroupBounds="boundedWithinViewport"
            disableAutoResizing={false}
          />
        </main>
        <footer className="status-bar">
          <span className={'status-dot ' + (state.busy ? 'busy' : '')} />
          <span className="status-message">{state.active?.label || state.validation.summary}</span>
          <span>{state.queue.filter((e) => e.status === 'Pending').length} queued</span>
          <span className="status-file" title={state.path || ''}>
            {state.path?.split(/[\\/]/).at(-1) || 'Unsaved recipe'}
          </span>
          <button onClick={() => reveal('console')}>Console</button>
        </footer>
      </div>
      <UtilityDialog tool={tool} onClose={() => setTool(null)} />
      {preferences && <Preferences onClose={() => setPreferences(false)} />}{' '}
      {question && (
        <Dialog
          title={question.title}
          onClose={() => answer(false)}
          footer={
            <>
              <button onClick={() => answer(false)}>Cancel</button>
              <button className="primary" onClick={() => answer(true)}>
                {question.confirm}
              </button>
            </>
          }
        >
          <p className="question-message">{question.message}</p>
        </Dialog>
      )}
      {working && (
        <div className="working-overlay" role="status">
          {working}
        </div>
      )}
    </ActionsContext.Provider>
  );
}
