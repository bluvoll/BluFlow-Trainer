// Exercise the real launch supervisor and a native window, beyond the offscreen UI smoke.
const assert = require('node:assert/strict');
const { spawn, spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { setTimeout: delay } = require('node:timers/promises');

async function verifyStartup() {
  const root = path.resolve(__dirname, '../..');
  const artifacts = path.join(root, '.artifacts/electron-startup');
  fs.mkdirSync(artifacts, { recursive: true });
  const profile = fs.mkdtempSync(path.join(artifacts, 'profile-'));
  const configs = path.join(profile, 'configs');
  fs.mkdirSync(configs);
  const child = spawn(
    process.execPath,
    [
      path.join(root, 'scripts/electron.cjs'),
      'start',
      '--test',
      '--test-visible',
      '--no-update-check',
      '--no-dependency-check',
      '--inspect=0',
      ...(process.platform === 'linux' && process.env.CI ? ['--no-sandbox'] : []),
    ],
    {
      cwd: root,
      // The supervisor is a console helper. Its Electron child must override this flag.
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: {
        ...process.env,
        BLUFLOW_TEST_PROFILE: profile,
        BLUFLOW_TEST_PREFS: path.join(profile, 'gui.json'),
        BLUFLOW_TEST_CONFIG_DIR: configs,
      },
    },
  );
  let stderr = '',
    launchError,
    socket,
    next = 0,
    exited = false;
  const pending = new Map();
  const exit = new Promise((resolve) =>
    child.once('exit', (code) => {
      exited = true;
      resolve(code);
    }),
  );
  child.once('error', (error) => {
    launchError = error;
  });
  child.stdout.resume();
  child.stderr.on('data', (data) => {
    stderr += data;
  });
  async function until(read, description) {
    const deadline = Date.now() + 45000;
    while (Date.now() < deadline) {
      if (launchError) throw launchError;
      if (exited) throw Error(`Launcher exited before ${description}: ${stderr}`);
      const result = await read();
      if (result) return result;
      await delay(100);
    }
    throw Error(`Timed out waiting for ${description}: ${stderr}`);
  }
  function evaluate(expression) {
    return new Promise((resolve, reject) => {
      const id = ++next;
      const timeout = setTimeout(() => {
        pending.delete(id);
        reject(Error('Startup inspector request timed out'));
      }, 10000);
      pending.set(id, (message) => {
        clearTimeout(timeout);
        if (message.error || message.result?.exceptionDetails)
          reject(Error(JSON.stringify(message.error || message.result.exceptionDetails)));
        else resolve(message.result.result.value);
      });
      socket.send(
        JSON.stringify({
          id,
          method: 'Runtime.evaluate',
          params: { expression, awaitPromise: true, returnByValue: true },
        }),
      );
    });
  }
  try {
    const url = await until(
      () => stderr.match(/Debugger listening on (ws:\/\/[^\s]+)/)?.[1],
      'Electron inspector',
    );
    socket = new WebSocket(url);
    await new Promise((resolve, reject) => {
      socket.onopen = resolve;
      socket.onerror = reject;
    });
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.id) {
        pending.get(message.id)?.(message);
        pending.delete(message.id);
      }
    };
    const loaded = `(async()=>{
      const win=process.mainModule.require('electron').BrowserWindow.getAllWindows()[0];
      if(!win||win.webContents.isLoading())return false;
      const ready=await win.webContents.executeJavaScript("!!document.querySelector('[data-testid=app]')");
      return ready?{visible:win.isVisible(),offscreen:win.webContents.isOffscreen()}:false;
    })()`;
    let latest;
    await until(async () => {
      latest = await evaluate(loaded);
      return latest?.visible;
    }, 'a visible training workspace');
    assert.equal(latest.offscreen, false, 'Verify a native window, not an offscreen renderer');
    await evaluate(`(async()=>{
      const win=process.mainModule.require('electron').BrowserWindow.getAllWindows()[0];
      const png=(await win.webContents.capturePage()).toPNG();
      process.mainModule.require('node:fs').writeFileSync(${JSON.stringify(path.join(artifacts, 'visible-workspace.png'))},png);
    })()`);
    console.log(
      'Native startup passed: the real launch script opens a visible training workspace.',
    );
  } finally {
    if (socket?.readyState === WebSocket.OPEN) {
      await evaluate(
        "process.mainModule.require('electron').BrowserWindow.getAllWindows().forEach(win=>win.close())",
      ).catch(() => {});
      socket.close();
    }
    let timeout;
    await Promise.race([
      exit,
      new Promise((resolve) => {
        timeout = setTimeout(resolve, 10000);
      }),
    ]);
    clearTimeout(timeout);
    if (!exited) {
      if (process.platform === 'win32')
        spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], {
          windowsHide: true,
          stdio: 'ignore',
        });
      else child.kill();
    }
  }
}

module.exports = { verifyStartup };
if (require.main === module)
  verifyStartup().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
