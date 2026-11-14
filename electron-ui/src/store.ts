import { useRef, useSyncExternalStore } from 'react';
import type { Bootstrap, DesktopEvent, State, Metrics, Flat } from './types';
type Store = {
  data: Bootstrap | null;
  state: State | null;
  metrics: Metrics;
  logs: { text: string; progress: boolean }[];
  consoleTotal: number;
  error: string;
  category: string;
  panel: string;
  paused: boolean;
  revision: number;
};
let current: Store = {
  data: null,
  state: null,
  metrics: { records: [], latest: {}, reports: [], startedAt: null, elapsed: null },
  logs: [],
  consoleTotal: 0,
  error: '',
  category: 'Training',
  panel: 'settings',
  paused: false,
  revision: 0,
};
const listeners = new Set<() => void>(),
  events = new Set<(event: DesktopEvent) => void>();
let pending: Record<string, { value: any; revision: number }> = {},
  timer: ReturnType<typeof setTimeout>,
  flight: Promise<void> | null = null,
  editRevision = 0;
function publish(patch: Partial<Store>) {
  current = { ...current, ...patch, revision: current.revision + 1 };
  for (const listener of listeners) listener();
}
export const state = () => current;
export function useStore<T = Store>(
  selector: (store: Store) => T = (value) => value as unknown as T,
) {
  const cache = useRef<{ value: T } | null>(null);
  const snapshot = () => {
    const value = selector(current),
      previous = cache.current?.value;
    if (
      cache.current &&
      (Object.is(value, previous) ||
        (value &&
          previous &&
          typeof value === 'object' &&
          typeof previous === 'object' &&
          Object.keys(value).length === Object.keys(previous).length &&
          Object.keys(value).every((key) =>
            Object.is((value as any)[key], (previous as any)[key]),
          )))
    )
      return previous as T;
    cache.current = { value };
    return value;
  };
  return useSyncExternalStore((listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }, snapshot);
}
function optimistic(flat: Flat) {
  return {
    ...flat,
    ...Object.fromEntries(Object.entries(pending).map(([key, edit]) => [key, edit.value])),
  };
}
export function accept(result: any) {
  const value: State | undefined = result?.state || (result?.flat && result);
  if (value) publish({ state: { ...value, flat: optimistic(value.flat) } });
  return result;
}
export function error(error: unknown) {
  publish({ error: error instanceof Error ? error.message : String(error) });
}
export function clearError() {
  publish({ error: '' });
}
export function chooseCategory(category: string) {
  publish({ category, panel: 'settings' });
}
export function choosePanel(panel: string) {
  publish({ panel });
}
export function subscribeEvents(callback: (event: DesktopEvent) => void) {
  events.add(callback);
  return () => {
    events.delete(callback);
  };
}
export function edit(key: string, value: any) {
  pending[key] = { value, revision: ++editRevision };
  if (current.state)
    publish({ state: { ...current.state, flat: { ...current.state.flat, [key]: value } } });
  clearTimeout(timer);
  timer = setTimeout(() => void flush().catch(error), 180);
}
export function flush(): Promise<void> {
  clearTimeout(timer);
  if (flight) return flight;
  flight = (async () => {
    while (Object.keys(pending).length) {
      const batch = { ...pending },
        patch = Object.fromEntries(Object.entries(batch).map(([key, item]) => [key, item.value]));
      const result = await window.bluflow.request<State>('config.edit', { patch });
      for (const [key, item] of Object.entries(batch))
        if (pending[key]?.revision === item.revision) delete pending[key];
      accept(result);
    }
  })().finally(() => {
    flight = null;
  });
  return flight;
}
export async function command(method: string, params: any = {}) {
  if (method === 'config.open') {
    clearTimeout(timer);
    pending = {};
    if (flight) await flight.catch(() => {});
  } else if (
    ![
      'job.stop',
      'metrics.control',
      'utility.run',
      'queue.start',
      'queue.add',
      'queue.edit',
      'preferences.set',
      'console.read',
      'console.text',
      'console.clear',
    ].includes(method)
  )
    await flush();
  return accept(await window.bluflow.request(method, params));
}
export async function initialize() {
  window.bluflow.onEvent((event) => {
    if (event.event === 'state') accept(event.state);
    else if (event.event === 'log') {
      const logs = [...current.logs];
      if (event.progress && logs.at(-1)?.progress) logs.pop();
      logs.push({ text: event.text, progress: event.progress });
      publish({ logs: logs.slice(-10000), consoleTotal: event.total ?? logs.length });
    } else if (event.event === 'console_clear') {
      publish({ logs: [], consoleTotal: 0 });
    } else if (event.event === 'metrics_reset') publish({ metrics: event.metrics });
    else if (event.event === 'metric')
      publish({
        metrics: {
          ...current.metrics,
          records: [...current.metrics.records, event.record].slice(-60000),
          latest: event.record,
        },
      });
    else if (event.event === 'metrics_compact')
      publish({ metrics: { ...current.metrics, records: event.records } });
    else if (event.event === 'report')
      publish({ metrics: { ...current.metrics, reports: event.reports } });
    else if (event.event === 'stage')
      publish({
        metrics: {
          ...current.metrics,
          latest: { ...current.metrics.latest, resolution_stage: event.stage },
        },
      });
    else if (event.event === 'metrics_finished')
      publish({ metrics: { ...current.metrics, startedAt: null, elapsed: event.elapsed } });
    else if (event.event === 'activate') choosePanel(event.panel);
    else if (event.event === 'backend_failure' || event.event === 'warning') error(event.message);
    for (const callback of events) callback(event);
  });
  const data = await window.bluflow.request<Bootstrap>('bootstrap');
  publish({
    data,
    state: data.state,
    metrics: data.metrics,
    logs: data.logs,
    consoleTotal: data.consoleTotal,
  });
}
export function setPaused(paused: boolean) {
  publish({ paused });
}
