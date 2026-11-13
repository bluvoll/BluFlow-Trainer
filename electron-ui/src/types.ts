export type Flat = Record<string, any>;
export type Field = {
  key: string;
  label: string;
  tooltip: string;
  inline: boolean;
  kind: string;
  args: any[];
  kwargs: Record<string, any>;
};
export type Page = [string, [string, string[]][]];
export type Metrics = {
  records: Flat[];
  latest: Flat;
  reports: string[];
  startedAt: number | null;
  elapsed: number | null;
};
export type QueueEntry = {
  id: number;
  source: string;
  snapshot: string;
  digest: string;
  status: string;
  error: string;
};
export type State = {
  previewGrid: [number, number];
  flat: Flat;
  path: string | null;
  configDir: string;
  preferences: Flat;
  presets: { path: string; name: string }[];
  gpus: { index: number; name: string }[];
  selectedGpus: number[];
  multiGpu: boolean;
  platform: string;
  busy: boolean;
  queueActive: boolean;
  active: { label: string; training: boolean; entryId: number | null } | null;
  queue: QueueEntry[];
  continueErrors: boolean;
  validation: {
    valid: boolean;
    error: string;
    summary: string;
    enabled: Record<string, boolean>;
    advisories: [string, string][];
    experimental: boolean;
    distribution: { probabilities: number[]; median: number; high_noise: number } | null;
    stages: Flat[] | null;
    windowsRepair: boolean;
  };
};
export type Bootstrap = {
  consoleTotal: number;
  fields: Record<string, Field>;
  pages: Page[];
  state: State;
  metrics: Metrics;
  logs: { text: string; progress: boolean }[];
  curriculumPreset: Flat[];
};
export type DesktopEvent = { event: string; [key: string]: any };
export type Presentation = { layout?: any; collapsed?: string[]; category?: string };
declare global {
  interface Window {
    bluflow: {
      request<T = any>(method: string, params?: any): Promise<T>;
      file(options: any): Promise<any>;
      preferences(patch?: Presentation): Promise<Presentation>;
      restart(options?: any): Promise<void>;
      snapshot(): Promise<Metrics>;
      ready(): void;
      onEvent(callback: (event: DesktopEvent) => void): () => void;
    };
  }
}
