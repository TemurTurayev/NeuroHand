import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

test('shell exposes the complete QLS route', async () => {
  const shell = await readFile(new URL('../src/shell.html', import.meta.url), 'utf8');
  assert.match(shell, /--violet:#6B4FA3/);
  assert.match(shell, /QLS Extension/);
  for (let number = 19; number <= 25; number += 1) {
    assert.match(shell, new RegExp(`data-go="ch-${number}"`));
    assert.match(shell, new RegExp(`'ch-${number}'`));
  }
});
