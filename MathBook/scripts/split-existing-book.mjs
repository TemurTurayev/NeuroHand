import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

import { splitBook } from './book-lib.mjs';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const bookDir = resolve(scriptDir, '..');
const sourcePath = join(bookDir, 'kniga.html');
const shellPath = join(bookDir, 'src', 'shell.html');
const chaptersDir = join(bookDir, 'src', 'chapters');
const force = process.argv.includes('--force');

await mkdir(chaptersDir, { recursive: true });
const existing = (await readdir(chaptersDir)).filter((name) => /^ch-\d+\.html$/.test(name));
if (existing.length > 0 && !force) {
  throw new Error('Chapter sources already exist; pass --force to replace them');
}

const source = await readFile(sourcePath, 'utf8');
const { shell, chapters } = splitBook(source);
await writeFile(shellPath, shell);
for (const [number, html] of chapters) {
  await writeFile(join(chaptersDir, `ch-${String(number).padStart(2, '0')}.html`), html);
}

console.log(`Extracted ${chapters.size} bilingual chapters into ${chaptersDir}`);
