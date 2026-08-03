import { readFile, readdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

import { buildBook, extractChapterPairs } from './book-lib.mjs';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const bookDir = resolve(scriptDir, '..');
const shellPath = join(bookDir, 'src', 'shell.html');
const chaptersDir = join(bookDir, 'src', 'chapters');
const outputPath = join(bookDir, 'kniga.html');

const shell = await readFile(shellPath, 'utf8');
const names = (await readdir(chaptersDir))
  .filter((name) => /^ch-\d+\.html$/.test(name))
  .sort();
const chapterFiles = await Promise.all(
  names.map(async (name) => ({
    number: Number(name.match(/\d+/)[0]),
    html: await readFile(join(chaptersDir, name), 'utf8'),
  })),
);
const output = buildBook(shell, chapterFiles);
extractChapterPairs(output);
await writeFile(outputPath, output);
console.log(`Built ${chapterFiles.length} bilingual chapters into ${outputPath}`);
