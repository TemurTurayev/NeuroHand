import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

import { validateBook } from './book-lib.mjs';

const optionIndex = process.argv.indexOf('--last-chapter');
if (optionIndex < 0 || !/^\d+$/.test(process.argv[optionIndex + 1] ?? '')) {
  throw new Error('Usage: validate-book.mjs --last-chapter N');
}
const lastChapter = Number(process.argv[optionIndex + 1]);
const scriptDir = dirname(fileURLToPath(import.meta.url));
const html = await readFile(resolve(scriptDir, '..', 'kniga.html'), 'utf8');
const report = validateBook(html, lastChapter);
console.log(
  `Validated ${report.chapterPairs} bilingual chapters and ${report.quizNames} quiz groups`,
);
