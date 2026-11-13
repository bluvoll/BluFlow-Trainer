const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '..');
function fingerprint() {
  const hash = crypto.createHash('sha256');
  function walk(relative) {
    for (const entry of fs
      .readdirSync(path.join(root, relative), { withFileTypes: true })
      .sort((a, b) => a.name.localeCompare(b.name))) {
      const p = path.join(relative, entry.name);
      if (entry.isDirectory()) walk(p);
      else hash.update(p).update(fs.readFileSync(path.join(root, p)));
    }
  }
  walk('src');
  for (const file of ['package.json', 'package-lock.json', 'index.html', 'vite.config.ts'])
    hash.update(file).update(fs.readFileSync(path.join(root, file)));
  return hash.digest('hex');
}
if (require.main === module)
  fs.writeFileSync(
    path.join(root, 'dist/build-stamp.json'),
    JSON.stringify({ fingerprint: fingerprint() }),
  );
module.exports = { fingerprint };
