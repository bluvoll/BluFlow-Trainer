const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');

function loadStore(request) {
  global.window = { bluflow: { request } };
  const filename = path.resolve(__dirname, '../src/store.ts');
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const compiledModule = new Module(filename, module);
  compiledModule.filename = filename;
  compiledModule.paths = Module._nodeModulePaths(path.dirname(filename));
  compiledModule._compile(compiled, filename);
  return compiledModule.exports;
}

test('Stop and independent actions remain available while a config edit is stuck or invalid', async () => {
  const calls = [];
  let rejectEdit;
  const store = loadStore((method) => {
    calls.push(method);
    return method === 'config.edit'
      ? new Promise((resolve, reject) => {
          rejectEdit = reject;
        })
      : Promise.resolve(true);
  });
  store.edit('train.run_name', 'invalid/name');
  const flight = store.flush();
  const failed = assert.rejects(flight, /Invalid name/);
  for (const method of [
    'job.stop',
    'metrics.control',
    'queue.start',
    'utility.run',
    'console.clear',
  ]) {
    let timeout;
    try {
      await Promise.race([
        store.command(method),
        new Promise((resolve, reject) => {
          timeout = setTimeout(() => reject(new Error(`${method} waited for config.edit`)), 1000);
        }),
      ]);
    } finally {
      clearTimeout(timeout);
    }
  }
  rejectEdit(new Error('Invalid name'));
  await failed;
  await store.command('job.stop');
  assert.equal(calls.filter((method) => method === 'config.edit').length, 1);
});

test('opening a preset discards an invalid pending draft', async () => {
  const calls = [];
  const store = loadStore(async (method) => {
    calls.push(method);
    if (method === 'config.edit') throw new Error('Invalid draft');
    return { flat: { 'train.run_name': 'valid-preset' } };
  });
  store.edit('train.run_name', 'invalid/name');
  await assert.rejects(store.flush(), /Invalid draft/);
  await store.command('config.open', { path: 'valid.toml' });
  await store.flush();
  assert.deepEqual(calls, ['config.edit', 'config.open']);
  assert.equal(store.state().state.flat['train.run_name'], 'valid-preset');
});

test('opening a preset waits for a failed in-flight draft and still opens it', async () => {
  let rejectEdit;
  const calls = [];
  const store = loadStore((method) => {
    calls.push(method);
    return method === 'config.edit'
      ? new Promise((resolve, reject) => {
          rejectEdit = reject;
        })
      : Promise.resolve({ flat: { 'train.run_name': 'opened' } });
  });
  store.edit('train.run_name', 'invalid/name');
  const failed = assert.rejects(store.flush(), /Invalid draft/);
  const open = store.command('config.open', { path: 'valid.toml' });
  assert.deepEqual(calls, ['config.edit']);
  rejectEdit(new Error('Invalid draft'));
  await failed;
  await open;
  await store.flush();
  assert.deepEqual(calls, ['config.edit', 'config.open']);
  assert.equal(store.state().state.flat['train.run_name'], 'opened');
});

test('block LR preview waits for the latest block edit to resolve', async () => {
  const resolvers = [];
  const store = loadStore(() => new Promise((resolve) => resolvers.push(resolve)));
  const response = (minimum) => ({
    flat: { 'block_lr.min_lr_mul': minimum },
    validation: { blockLRPreview: { multipliers: [minimum], error: '' } },
  });
  const preview = () => store.state().state.validation.blockLRPreview;
  store.accept(response(0.1));
  assert.deepEqual(preview().multipliers, [0.1]);
  store.edit('block_lr.min_lr_mul', 0.2);
  const flight = store.flush();
  assert.equal(preview(), null);
  store.edit('block_lr.min_lr_mul', 0.3);
  resolvers[0](response(0.2));
  await Promise.resolve();
  assert.equal(preview(), null);
  assert.equal(store.state().state.flat['block_lr.min_lr_mul'], 0.3);
  resolvers[1](response(0.3));
  await flight;
  await store.flush();
  assert.deepEqual(preview().multipliers, [0.3]);
});

test('failed config.edit removes failed pending patch instead of poisoning subsequent flushes', async () => {
  const patches = [];
  const store = loadStore(async (method, params) => {
    if (method === 'config.edit') {
      patches.push(params.patch);
      if ('bad.key' in params.patch) throw new Error('Bad key');
      return { flat: { ...params.patch } };
    }
    return true;
  });
  store.edit('bad.key', 1);
  await assert.rejects(store.flush(), /Bad key/);
  store.edit('good.key', 2);
  await store.flush();
  assert.equal(patches.length, 2);
  assert.deepEqual(patches[1], { 'good.key': 2 });
});
