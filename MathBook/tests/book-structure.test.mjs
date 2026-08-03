import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
  extractChapterPairs,
  sha256,
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
