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

test('legacy chapters remain unchanged inside the 26-chapter publication', async () => {
  const legacy = (
    await Promise.all(
      Array.from({ length: 19 }, (_, index) =>
        readFile(
          new URL(
            `../src/chapters/ch-${String(index).padStart(2, '0')}.html`,
            import.meta.url,
          ),
          'utf8',
        ),
      ),
    )
  ).join('');
  assert.equal(
    sha256(legacy),
    'dc985a393b7f0243447d9626cdc089318aec537d65f9dbe9a26e2589aa923c36',
  );
  const pairs = extractChapterPairs(current);
  assert.equal(pairs.length, 26);
  assert.deepEqual(
    pairs.map((pair) => pair.number),
    Array.from({ length: 26 }, (_, index) => index),
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
