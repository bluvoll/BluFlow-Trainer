const {
  app,
  BrowserWindow,
  ipcMain,
  dialog,
  protocol,
  net,
  Menu,
  powerSaveBlocker,
  screen,
} = require('electron');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { randomUUID } = require('node:crypto');
const { BackendPipe } = require('./pipe.cjs');
const root = path.resolve(__dirname, '../..'),
  dist = path.resolve(__dirname, '../dist');
const test = process.argv.includes('--test');
const headless = test && !process.argv.includes('--test-visible');
const args = process.argv.slice(2);
let window,
  child,
  pipe,
  closing = false,
  presentation = {},
  blocker,
  lastState;
const captures = new Map();
const pendingCaptures = new Set();
app.setName('BluFlow-Trainer Electron');
if (test) {
  if (!process.env.BLUFLOW_TEST_PROFILE) throw Error('Tests require BLUFLOW_TEST_PROFILE');
  app.setPath('userData', path.resolve(process.env.BLUFLOW_TEST_PROFILE));
  if (headless) app.disableHardwareAcceleration();
} else app.setPath('userData', path.join(app.getPath('appData'), 'BluFlow-Trainer-Electron'));
protocol.registerSchemesAsPrivileged([
  { scheme: 'bluflow-app', privileges: { standard: true, secure: true, supportFetchAPI: true } },
]);
function send(event) {
  if (window && !window.isDestroyed()) window.webContents.send('bluflow:event', event);
}
function rememberState(state) {
  if (!state) return;
  lastState = state;
  if (state.busy && state.active?.training && blocker === undefined)
    blocker = powerSaveBlocker.start('prevent-app-suspension');
  if ((!state.busy || !state.active?.training) && blocker !== undefined) {
    powerSaveBlocker.stop(blocker);
    blocker = undefined;
  }
}
function trusted(event) {
  if (event.sender !== window?.webContents) throw Error('Untrusted desktop caller');
}
const allowed = new Set([
  'bootstrap',
  'state',
  'config.open',
  'config.edit',
  'config.save',
  'preferences.set',
  'gpus.set',
  'preview.timestep',
  'queue.add',
  'queue.edit',
  'queue.start',
  'job.start',
  'job.stop',
  'metrics.get',
  'console.read',
  'console.text',
  'console.clear',
  'metrics.control',
  'utility.run',
  'startup.check',
  'startup.update',
]);
ipcMain.handle('bluflow:request', async (event, method, params) => {
  trusted(event);
  if (!allowed.has(method)) throw Error('Unknown desktop operation');
  if (method === 'startup.check')
    params = {
      ...params,
      skipUpdate: args.includes('--no-update-check'),
      skipDependencies: args.includes('--no-dependency-check'),
    };
  const result = await pipe.call(method, params);
  rememberState(result?.state || (result?.flat && result));
  return result;
});
ipcMain.handle('bluflow:presentation', (event, patch) => {
  trusted(event);
  if (patch !== undefined) {
    if (
      !patch ||
      typeof patch !== 'object' ||
      Object.keys(patch).some((k) => !['layout', 'collapsed', 'category'].includes(k))
    )
      throw Error('Unknown workspace preference');
    if (JSON.stringify(patch).length > 2_000_000) throw Error('Workspace layout is too large');
    presentation = { ...presentation, ...patch };
    savePresentation();
  }
  return presentation;
});
ipcMain.handle('bluflow:file', async (event, options = {}) => {
  trusted(event);
  const { kind = 'open', multiple = false, defaultPath } = options;
  const filters =
    kind === 'toml' || kind === 'save-toml'
      ? [{ name: 'TOML configs', extensions: ['toml'] }]
      : [
          { name: 'Model files', extensions: ['safetensors', 'pt', 'bin'] },
          { name: 'All files', extensions: ['*'] },
        ];
  if (kind.startsWith('save')) {
    const result = await dialog.showSaveDialog(window, { defaultPath, filters });
    return result.canceled ? null : result.filePath;
  }
  const result = await dialog.showOpenDialog(window, {
    defaultPath,
    filters,
    properties:
      kind === 'folder'
        ? ['openDirectory']
        : ['openFile', ...(multiple ? ['multiSelections'] : [])],
  });
  return result.canceled ? null : multiple ? result.filePaths : result.filePaths[0];
});
ipcMain.handle('bluflow:restart', async (event, options = {}) => {
  trusted(event);
  if (lastState?.busy) throw Error('Stop the current job before restarting');
  closing = true;
  await stopBackend();
  app.exit(options.refreshDependencies && !args.includes('--no-dependency-check') ? 75 : 76);
});
ipcMain.handle('bluflow:snapshot', (event) => {
  const item = captures.get(event.sender.id);
  if (!item) throw Error('Unknown metrics capture');
  return item.metrics;
});
ipcMain.on('bluflow:ready', (event) => {
  const capture = captures.get(event.sender.id);
  if (capture) {
    capture.ready();
    return;
  }
  if (event.sender === window?.webContents && !headless) window.show();
});
function presentationPath() {
  return path.join(app.getPath('userData'), 'workspace.json');
}
function savePresentation() {
  try {
    fs.mkdirSync(app.getPath('userData'), { recursive: true });
    const target = presentationPath(),
      temporary = target + '.tmp';
    fs.writeFileSync(temporary, JSON.stringify(presentation));
    fs.renameSync(temporary, target);
  } catch (error) {
    send({ event: 'warning', message: 'Workspace could not be saved: ' + error.message });
  }
}
function configure(win) {
  win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  win.webContents.on('will-navigate', (event, url) => {
    if (!url.startsWith('bluflow-app://ui/')) event.preventDefault();
  });
}
async function captureMetrics(event) {
  const capture = new BrowserWindow({
    show: false,
    width: 1280,
    height: 900,
    useContentSize: true,
    backgroundColor: '#101116',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      offscreen: true,
      backgroundThrottling: false,
    },
  });
  configure(capture);
  let timeout;
  try {
    const ready = new Promise((resolve, reject) => {
      captures.set(capture.webContents.id, { metrics: event.metrics, ready: resolve });
      timeout = setTimeout(() => reject(Error('Metrics capture timed out')), 15000);
    });
    await capture.loadURL('bluflow-app://ui/index.html?snapshot=' + randomUUID());
    await ready;
    const png = (await capture.webContents.capturePage()).toPNG();
    if (!png.length) throw Error('Metrics capture was empty');
    fs.mkdirSync(path.dirname(event.path), { recursive: true });
    const temporary = event.path + '.' + randomUUID() + '.tmp';
    try {
      fs.writeFileSync(temporary, png);
      fs.renameSync(temporary, event.path);
    } finally {
      if (fs.existsSync(temporary)) fs.unlinkSync(temporary);
    }
    await pipe.call('metrics.exportResult', { path: event.path });
  } catch (error) {
    await pipe.call('metrics.exportResult', { error: error.message }).catch(() => {});
  } finally {
    clearTimeout(timeout);
    captures.delete(capture.webContents.id);
    capture.destroy();
  }
}
function queueMetricsCapture(event) {
  const task = captureMetrics(event);
  pendingCaptures.add(task);
  task.finally(() => pendingCaptures.delete(task));
  return task;
}
async function stopBackend() {
  if (blocker !== undefined) {
    powerSaveBlocker.stop(blocker);
    blocker = undefined;
  }
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  await pipe.call('job.stop').catch(() => {});
  await Promise.allSettled([...pendingCaptures]);
  pipe.stop();
  await new Promise((resolve) => {
    const timer = setTimeout(() => {
      if (child.exitCode === null) child.kill();
      resolve();
    }, 25000);
    child.once('exit', () => {
      clearTimeout(timer);
      resolve();
    });
  });
}
app
  .whenReady()
  .then(async () => {
    Menu.setApplicationMenu(null);
    protocol.handle('bluflow-app', (request) => {
      const url = new URL(request.url);
      if (url.hostname !== 'ui') return new Response('Not found', { status: 404 });
      const target = path.resolve(dist, '.' + decodeURIComponent(url.pathname));
      if (!target.startsWith(dist + path.sep)) return new Response('Not found', { status: 404 });
      return net.fetch(pathToFileURL(target).toString());
    });
    try {
      presentation = JSON.parse(fs.readFileSync(presentationPath(), 'utf8'));
    } catch {}
    const python =
      process.env.BLUFLOW_PYTHON ||
      path.join(root, 'venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
    const backendArgs = ['-u', '-m', 'trainer.electron_backend'];
    const at = args.indexOf('--config-dir');
    if (at >= 0 && args[at + 1]) backendArgs.push('--config-dir', args[at + 1]);
    if (test) {
      if (!process.env.BLUFLOW_TEST_PREFS || !process.env.BLUFLOW_TEST_CONFIG_DIR)
        throw Error('Tests require isolated preferences and config folders');
      backendArgs.push(
        '--preferences',
        process.env.BLUFLOW_TEST_PREFS,
        '--config-dir',
        process.env.BLUFLOW_TEST_CONFIG_DIR,
      );
    }
    child = spawn(python, backendArgs, {
      cwd: root,
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' },
    });
    pipe = new BackendPipe(
      child,
      (event) => {
        if (event.event === 'state') rememberState(event.state);
        if (event.event === 'metrics_snapshot') void queueMetricsCapture(event);
        else send(event);
      },
      (error) => {
        if (!closing) send({ event: 'backend_failure', message: error.message });
      },
    );
    child.stderr.on('data', (data) => {
      process.stderr.write(data);
    });
    const display = screen.getPrimaryDisplay().workAreaSize;
    const bounds = presentation.bounds || {};
    const width = Math.max(1050, Math.min(Number(bounds.width) || 1600, display.width));
    const height = Math.max(700, Math.min(Number(bounds.height) || 1000, display.height));
    window = new BrowserWindow({
      width,
      height,
      minWidth: 1050,
      minHeight: 700,
      title: 'BluFlow-Trainer',
      show: false,
      backgroundColor: '#101116',
      webPreferences: {
        preload: path.join(__dirname, 'preload.cjs'),
        contextIsolation: true,
        sandbox: true,
        nodeIntegration: false,
        backgroundThrottling: false,
        offscreen: headless,
      },
    });
    configure(window);
    window.on('close', (event) => {
      if (closing) return;
      event.preventDefault();
      void (async () => {
        if (lastState?.busy) {
          const result = await dialog.showMessageBox(window, {
            type: 'question',
            title: 'Training is running',
            message: 'Stop the running process and quit?',
            buttons: ['Keep running', 'Stop and quit'],
            defaultId: 0,
            cancelId: 0,
          });
          if (result.response !== 1) return;
        }
        closing = true;
        presentation.bounds = window.getBounds();
        savePresentation();
        await stopBackend();
        app.quit();
      })();
    });
    await window.loadURL('bluflow-app://ui/index.html' + (test ? '?test=1' : ''));
  })
  .catch((error) => {
    console.error(error);
    if (!test) dialog.showErrorBox('BluFlow-Trainer could not start', error.message);
    closing = true;
    void stopBackend().finally(() => app.exit(1));
  });
app.on('window-all-closed', () => app.quit());
app.on('before-quit', (event) => {
  if (!closing && window && !window.isDestroyed()) {
    event.preventDefault();
    window.close();
  }
});
if (test) app.__bluflowCaptureMetrics = queueMetricsCapture;
