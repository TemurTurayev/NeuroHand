import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
  buildBook,
  extractChapterPairs,
  sha256,
  splitBook,
  validateBook,
} from '../scripts/book-lib.mjs';

const current = await readFile(new URL('../kniga.html', import.meta.url), 'utf8');

test('current publication is the reviewed 19-chapter baseline', () => {
  assert.equal(
    sha256(current),
    '69d9b7392df0288866f086bff1a583f7d8b6c084657dc587eb964f6ba2247bda',
  );
  const pairs = extractChapterPairs(current);
  assert.equal(pairs.length, 19);
  assert.deepEqual(
    pairs.map((pair) => pair.number),
    Array.from({ length: 19 }, (_, index) => index),
  );
  assert.match(pairs[12].ru, /Статистика: искусство не быть обманутым/);
  assert.match(pairs[13].en, /Probability/);
});

test('split and build round-trip preserves the current publication', () => {
  const { shell, chapters } = splitBook(current);
  const rebuilt = buildBook(
    shell,
    [...chapters].map(([number, html]) => ({ number, html })),
  );
  assert.equal(sha256(rebuilt), sha256(current));
});

test('validator rejects a missing bilingual twin', () => {
  assert.throws(
    () =>
      validateBook(
        '<main><section class="chapter" id="ch-0" data-ch="0" data-lang="ru"></section></main>',
        0,
      ),
    /twin/i,
  );
});

test('validator rejects duplicate quiz radio names across chapters', () => {
  const chapter = (number, lang) =>
    `<section class="chapter" id="ch-${number}${lang === 'en' ? '-en' : ''}" data-ch="${number}" data-lang="${lang}"><input type="radio" name="same-name"></section>`;
  const html = `<main>${chapter(0, 'ru')}${chapter(0, 'en')}</main>`;
  assert.throws(() => validateBook(html, 0), /duplicate quiz radio name/i);
});

test('compatibility entry points preserve chapter hashes', async () => {
  for (const name of ['frame.html', 'index.html']) {
    const html = await readFile(new URL(`../${name}`, import.meta.url), 'utf8');
    assert.match(html, /kniga\.html['"]\s*\+\s*location\.hash/);
  }
});
