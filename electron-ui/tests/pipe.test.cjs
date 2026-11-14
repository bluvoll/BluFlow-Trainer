const { test } = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { PassThrough } = require('node:stream');
const { BackendPipe } = require('../electron/pipe.cjs');
function child() {
  const c = new EventEmitter();
  c.stdin = new PassThrough();
  c.stdout = new PassThrough();
  return c;
}
test('RPC handles fragmented lines, events, out-of-order responses and backend errors', async () => {
  const c = child(),
    events = [],
    pipe = new BackendPipe(c, (event) => events.push(event));
  const first = pipe.call('state'),
    second = pipe.call('config.save');
  c.stdout.write('{"event":"log","text":"日本語"}\n{"id":2,"result":');
  c.stdout.write('{"path":"copy.toml"}}\n{"id":1,"result":{"busy":false}}\n');
  assert.deepEqual(await second, { path: 'copy.toml' });
  assert.deepEqual(await first, { busy: false });
  assert.equal(events[0].text, '日本語');
  const failing = pipe.call('job.start');
  c.stdout.write('{"id":3,"error":{"message":"Invalid configuration"}}\n');
  await assert.rejects(failing, /Invalid configuration/);
  pipe.stop();
});
test('backend exit rejects pending and future requests', async () => {
  const c = child(),
    errors = [],
    pipe = new BackendPipe(
      c,
      () => {},
      (error) => errors.push(error),
    );
  const pending = pipe.call('state');
  c.emit('exit', 7);
  await assert.rejects(pending, /exit 7/);
  await assert.rejects(pipe.call('state'), /unavailable/);
  assert.equal(errors.length, 1);
  assert.equal(pipe.pending.size, 0);
});
test('stream errors cannot leave promises pending', async () => {
  const c = child(),
    pipe = new BackendPipe(c);
  const pending = pipe.call('state');
  c.stdin.emit('error', Error('Pipe closed'));
  await assert.rejects(pending, /Pipe closed/);
  assert.equal(pipe.pending.size, 0);
});
test('stdout read errors reject pending calls instead of crashing Electron', async () => {
  const c = child(),
    pipe = new BackendPipe(c);
  const pending = pipe.call('state');
  c.stdout.emit('error', Error('Read failed'));
  await assert.rejects(pending, /Read failed/);
});
test('shutdown rejects outstanding calls and writes a shutdown request', async () => {
  const c = child(),
    pipe = new BackendPipe(c);
  let written = '';
  c.stdin.on('data', (data) => (written += data));
  const pending = pipe.call('state');
  pipe.stop();
  await assert.rejects(pending, /closing/);
  assert.equal(JSON.parse(written.trim().split('\n').at(-1)).method, 'shutdown');
});
