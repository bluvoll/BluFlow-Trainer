const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'),
  path = require('node:path'),
  os = require('node:os');
const { spawnSync } = require('node:child_process');
const root = path.resolve(__dirname, '../..'),
  script = path.join(root, 'scripts/electron.cjs');
function launch(action, sequence = [0], extra = {}) {
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), 'bluflow-launch-contract-'));
  const trace = path.join(folder, 'trace.json'),
    mock = path.join(folder, 'mock.cjs');
  fs.writeFileSync(
    mock,
    `
const fs=require('node:fs');const calls=[];const codes=${JSON.stringify(sequence)};
require('node:child_process').spawnSync=(command,args,options)=>{
 calls.push({command,args,cwd:options.cwd,windowsHide:options.windowsHide,env:options.env?{ELECTRON_RUN_AS_NODE:options.env.ELECTRON_RUN_AS_NODE,BLUFLOW_PYTHON:options.env.BLUFLOW_PYTHON}:null});
 fs.writeFileSync(${JSON.stringify(trace)},JSON.stringify(calls));
 return {status:args[0]===${JSON.stringify(path.join(root, 'electron-ui'))}?codes.shift()??0:0};
};`,
  );
  const result = spawnSync(
    process.execPath,
    ['--require', mock, script, action, '--config-dir', 'folder with spaces/日本語'],
    {
      cwd: root,
      encoding: 'utf8',
      env: {
        ...process.env,
        BLUFLOW_PYTHON: process.execPath,
        ELECTRON_RUN_AS_NODE: '1',
        ...extra,
      },
    },
  );
  return { result, calls: fs.existsSync(trace) ? JSON.parse(fs.readFileSync(trace, 'utf8')) : [] };
}
test('launcher forwards arguments and isolates Electron from Node-only environment', () => {
  const { result, calls } = launch('start');
  assert.equal(result.status, 0, result.stderr);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].cwd, root);
  assert.deepEqual(calls[0].args, [
    path.join(root, 'electron-ui'),
    '--config-dir',
    'folder with spaces/日本語',
  ]);
  assert.equal(calls[0].env.ELECTRON_RUN_AS_NODE, undefined);
  assert.equal(calls[0].env.BLUFLOW_PYTHON, process.execPath);
  assert.equal(
    calls[0].windowsHide,
    false,
    'The interactive Electron window must be allowed to show',
  );
});
test('setup reuses the interpreter and only installs Node dependencies', () => {
  const { result, calls } = launch('setup');
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(calls[0].args, ['-m', 'trainer.electron_backend', '--check']);
  assert.ok(calls.some((c) => c.args.includes('ci')));
  assert.ok(calls.some((c) => c.args.includes('build')));
  assert.ok(!calls.some((c) => c.args.includes('pip')));
});
test('missing trainer environment fails before installing anything', () => {
  const { result, calls } = launch('setup', [0], {
    BLUFLOW_PYTHON: path.join(root, '.artifacts/nonexistent/python'),
  });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /installer|install\.bat|install\.sh/);
  assert.deepEqual(calls, []);
});
test('approved restart closes Electron before Python refresh and UI rebuild', () => {
  const { result, calls } = launch('start', [75, 0]);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(calls[0].args[0], path.join(root, 'electron-ui'));
  assert.deepEqual(calls[1].args, ['-m', 'trainer.tools.update_dependencies', '--root', root]);
  assert.ok(calls[2].args.includes('ci'));
  assert.ok(calls[3].args[0].endsWith('install.js'));
  assert.ok(calls[4].args.includes('build'));
  assert.ok(calls[5].args.includes('--no-update-check'));
});
