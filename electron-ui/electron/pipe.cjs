const readline = require('node:readline');
class BackendPipe {
  constructor(child, onEvent = () => {}, onFailure = () => {}) {
    this.child = child;
    this.next = 0;
    this.pending = new Map();
    this.closed = false;
    this.lines = readline.createInterface({ input: child.stdout });
    this.lines.on('line', (line) => {
      try {
        const item = JSON.parse(line);
        if (item.event) {
          onEvent(item);
          return;
        }
        const pending = this.pending.get(item.id);
        if (!pending) return;
        this.pending.delete(item.id);
        item.error ? pending.reject(new Error(item.error.message)) : pending.resolve(item.result);
      } catch (error) {
        this.fail(new Error('Invalid Python response: ' + error.message));
      }
    });
    const fail = (error) => {
      if (!this.closed) {
        this.fail(error);
        onFailure(error);
      }
    };
    child.stdin.on('error', fail);
    child.stdout.on('error', fail);
    this.lines.on('error', fail);
    child.on('error', fail);
    child.on('exit', (code) =>
      fail(new Error('Python backend stopped (exit ' + code + '). Relaunch BluFlow-Trainer.')),
    );
  }
  fail(error) {
    this.closed = true;
    for (const task of this.pending.values()) task.reject(error);
    this.pending.clear();
  }
  call(method, params = {}) {
    if (this.closed)
      return Promise.reject(new Error('Python backend is unavailable. Relaunch BluFlow-Trainer.'));
    const id = ++this.next;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      try {
        this.child.stdin.write(JSON.stringify({ id, method, params }) + '\n', (error) => {
          if (error) {
            this.pending.delete(id);
            reject(error);
          }
        });
      } catch (error) {
        this.pending.delete(id);
        reject(error);
      }
    });
  }
  stop() {
    if (this.closed) return;
    try {
      this.child.stdin.end(JSON.stringify({ method: 'shutdown' }) + '\n');
    } catch {}
    this.fail(new Error('BluFlow-Trainer is closing.'));
  }
}
module.exports = { BackendPipe };
