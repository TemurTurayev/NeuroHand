# MathBook QLS Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the bilingual MathBook with seven interactive QLS mathematics chapters while preserving all existing content, URLs, and progress.

**Architecture:** Migrate the 2.27 MB monolithic publication into a source shell plus one bilingual fragment per chapter, then use dependency-free Node.js scripts to build and validate a single deployable `kniga.html`. Keep browser state and routing compatible, turn `frame.html` into a hash-preserving redirect, and add chapters 19–25 through the existing HTML component vocabulary.

**Tech Stack:** HTML5, CSS, vanilla JavaScript, inline SVG, Node.js 22 standard library (`node:test`, `fs`, `crypto`), static Python/Jupyter examples, GitHub Pages.

## Global Constraints

- Preserve the educational content of existing chapters 0–18 byte-for-byte inside their extracted section fragments.
- Preserve the `mikroskop-v1` local-storage namespace and existing `done`, `quiz`, `last`, and `lang` fields.
- The generated `MathBook/kniga.html` must contain exactly 26 Russian and 26 English chapter sections, numbered 0–25.
- The public book must remain a single self-contained HTML file with no runtime package, CDN, backend, or in-browser Python dependency.
- Each new chapter must contain at least three worked examples, four interactive quizzes, six exercises with hints and solutions, one deliberate-error or flashback exercise, one boss, one cheat sheet, one Feynman task, and one copyable Python mini-lab.
- Russian and English chapter pairs must be conceptually and numerically equivalent.
- Medical content is educational, not clinical guidance; prediction, association, and causation must remain distinct.
- Build and validation commands must work with Node.js 22 without `npm install`.

---

## File Map

### New source and tooling files

- `MathBook/src/shell.html` — the application shell, styles, course map, router, state, and the `<!--CHAPTERS-->` insertion marker.
- `MathBook/src/chapters/ch-00.html` … `ch-25.html` — one file per chapter, containing the Russian section followed by its English twin.
- `MathBook/scripts/book-lib.mjs` — reusable extraction, ordering, building, hashing, and validation functions.
- `MathBook/scripts/split-existing-book.mjs` — one-time CLI migration from the current monolith.
- `MathBook/scripts/build-book.mjs` — deterministic CLI builder for `kniga.html`.
- `MathBook/scripts/validate-book.mjs` — CLI structural validator.
- `MathBook/scripts/validate-python-examples.py` — syntax-checks all embedded Python examples.
- `MathBook/tests/book-structure.test.mjs` — source/build/entry-point tests.
- `MathBook/tests/chapter-content.test.mjs` — curriculum and bilingual-parity tests.
- `MathBook/tests/browser-smoke.mjs` — lightweight browser-independent checks of router and redirect markup; interactive browser testing remains a final manual step.
- `MathBook/README.md` — editing, building, validating, and publishing instructions.

### Modified generated/public files

- `MathBook/kniga.html` — generated complete publication.
- `MathBook/frame.html` — compatibility redirect preserving the hash.
- `MathBook/index.html` — canonical redirect preserving the hash.

---

### Task 1: Lock the Current Book with Baseline Tests

**Files:**
- Create: `MathBook/tests/book-structure.test.mjs`
- Create: `MathBook/scripts/book-lib.mjs`

**Interfaces:**
- Produces: `extractChapterPairs(html: string): Array<{number: number, ru: string, en: string, combined: string}>`
- Produces: `sha256(text: string): string`
- Produces: `assertContiguousNumbers(numbers: number[], first: number, last: number): void`

- [ ] **Step 1: Write the failing baseline test**

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { extractChapterPairs, sha256 } from '../scripts/book-lib.mjs';

const current = await readFile(new URL('../kniga.html', import.meta.url), 'utf8');

test('current publication is the reviewed 19-chapter baseline', () => {
  assert.equal(sha256(current), '69d9b7392df0288866f086bff1a583f7d8b6c084657dc587eb964f6ba2247bda');
  const pairs = extractChapterPairs(current);
  assert.equal(pairs.length, 19);
  assert.deepEqual(pairs.map(pair => pair.number), Array.from({length: 19}, (_, i) => i));
  assert.match(pairs[12].ru, /Статистика: искусство не быть обманутым/);
  assert.match(pairs[13].en, /Probability/);
});
```

- [ ] **Step 2: Run the test and verify the missing module failure**

Run: `node --test MathBook/tests/book-structure.test.mjs`

Expected: FAIL with `ERR_MODULE_NOT_FOUND` for `book-lib.mjs`.

- [ ] **Step 3: Implement exact section-pair extraction**

Use a section scanner that locates `<section class="chapter" ...>` boundaries and pairs `id="ch-N"` with `id="ch-N-en"`. Do not parse arbitrary HTML with one greedy regex.

```js
import { createHash } from 'node:crypto';

export function sha256(text) {
  return createHash('sha256').update(text).digest('hex');
}

export function chapterSections(html) {
  const starts = [...html.matchAll(/<section class="chapter" id="ch-(\d+)(-en)?"[^>]*>/g)];
  return starts.map((match, index) => ({
    number: Number(match[1]),
    lang: match[2] ? 'en' : 'ru',
    html: html.slice(match.index, starts[index + 1]?.index ?? html.indexOf('</main>', match.index)),
  }));
}

export function extractChapterPairs(html) {
  const grouped = new Map();
  for (const section of chapterSections(html)) {
    const pair = grouped.get(section.number) ?? {number: section.number};
    pair[section.lang] = section.html;
    grouped.set(section.number, pair);
  }
  return [...grouped.values()].sort((a, b) => a.number - b.number).map(pair => {
    if (!pair.ru || !pair.en) throw new Error(`Missing bilingual twin for chapter ${pair.number}`);
    return {...pair, combined: pair.ru + pair.en};
  });
}
```

- [ ] **Step 4: Run the baseline test**

Run: `node --test MathBook/tests/book-structure.test.mjs`

Expected: PASS, one test.

- [ ] **Step 5: Commit the baseline lock**

```bash
git add MathBook/scripts/book-lib.mjs MathBook/tests/book-structure.test.mjs
git commit -m "test: lock MathBook publication baseline"
```

---

### Task 2: Split and Rebuild the Existing Monolith Without Content Drift

**Files:**
- Create: `MathBook/scripts/split-existing-book.mjs`
- Create: `MathBook/scripts/build-book.mjs`
- Create: `MathBook/src/shell.html`
- Create: `MathBook/src/chapters/ch-00.html` … `ch-18.html`
- Modify: `MathBook/tests/book-structure.test.mjs`

**Interfaces:**
- Consumes: `extractChapterPairs`, `sha256`
- Produces: `splitBook(sourceHtml: string): {shell: string, chapters: Map<number, string>}`
- Produces: `buildBook(shell: string, chapterFiles: Array<{number: number, html: string}>): string`

- [ ] **Step 1: Add a failing round-trip test**

```js
import { splitBook, buildBook } from '../scripts/book-lib.mjs';

test('split and build round-trip preserves the current publication', () => {
  const {shell, chapters} = splitBook(current);
  const rebuilt = buildBook(shell, [...chapters].map(([number, html]) => ({number, html})));
  assert.equal(sha256(rebuilt), sha256(current));
});
```

- [ ] **Step 2: Run the round-trip test**

Run: `node --test MathBook/tests/book-structure.test.mjs`

Expected: FAIL because `splitBook` and `buildBook` are not exported.

- [ ] **Step 3: Implement deterministic split/build functions**

`splitBook` must replace the exact contiguous range from the first Russian chapter section through the end of the English chapter 18 section with `<!--CHAPTERS-->`. `buildBook` must sort chapters numerically and replace the marker with their unmodified concatenation.

```js
export function buildBook(shell, chapterFiles) {
  const marker = '<!--CHAPTERS-->';
  if ((shell.match(/<!--CHAPTERS-->/g) ?? []).length !== 1) {
    throw new Error('Shell must contain exactly one CHAPTERS marker');
  }
  const body = [...chapterFiles]
    .sort((a, b) => a.number - b.number)
    .map(chapter => chapter.html)
    .join('');
  return shell.replace(marker, body);
}
```

- [ ] **Step 4: Implement the migration and builder CLIs**

`split-existing-book.mjs` reads `MathBook/kniga.html`, refuses to overwrite non-empty `src/chapters` without `--force`, writes 19 fragments named with two digits, and writes `src/shell.html`.

`build-book.mjs` reads all `ch-NN.html` fragments, calls `buildBook`, and writes `MathBook/kniga.html` only after validation succeeds.

- [ ] **Step 5: Run the migration and prove byte equivalence**

Run:

```bash
node MathBook/scripts/split-existing-book.mjs
node MathBook/scripts/build-book.mjs
shasum -a 256 MathBook/kniga.html
```

Expected SHA-256: `69d9b7392df0288866f086bff1a583f7d8b6c084657dc587eb964f6ba2247bda`.

- [ ] **Step 6: Run tests and inspect the generated diff**

Run:

```bash
node --test MathBook/tests/book-structure.test.mjs
git diff --exit-code -- MathBook/kniga.html
```

Expected: tests PASS and no diff for `kniga.html`.

- [ ] **Step 7: Commit modular sources**

```bash
git add MathBook/src MathBook/scripts MathBook/tests/book-structure.test.mjs
git commit -m "refactor: modularize MathBook sources"
```

---

### Task 3: Add Structural Validation and Repair Public Entry Points

**Files:**
- Create: `MathBook/scripts/validate-book.mjs`
- Modify: `MathBook/scripts/book-lib.mjs`
- Modify: `MathBook/frame.html`
- Modify: `MathBook/index.html`
- Modify: `MathBook/tests/book-structure.test.mjs`

**Interfaces:**
- Produces: `validateBook(html: string, expectedLastChapter: number): {chapterPairs: number, quizNames: number}`
- Produces redirect contract: `location.replace('kniga.html' + location.hash)`

- [ ] **Step 1: Write failing validator and redirect tests**

```js
import { validateBook } from '../scripts/book-lib.mjs';

test('validator rejects missing English twins and duplicate quiz names', () => {
  assert.throws(() => validateBook('<section class="chapter" id="ch-0" data-ch="0"></section>', 0), /twin/);
});

test('compatibility entry points preserve chapter hashes', async () => {
  for (const name of ['frame.html', 'index.html']) {
    const html = await readFile(new URL(`../${name}`, import.meta.url), 'utf8');
    assert.match(html, /kniga\.html['"]\s*\+\s*location\.hash/);
  }
});
```

- [ ] **Step 2: Run tests and verify failures**

Run: `node --test MathBook/tests/book-structure.test.mjs`

Expected: FAIL for missing `validateBook` and old redirects.

- [ ] **Step 3: Implement structural validation**

Validate chapter numbering and twins, unique section IDs, unique quiz radio names, exactly one chapter insertion marker in the shell, valid internal `#ch-N` links, and absence of external `<script src>` or stylesheet dependencies.

```js
export function assertContiguousNumbers(numbers, first, last) {
  const expected = Array.from({length: last - first + 1}, (_, i) => first + i);
  if (JSON.stringify(numbers) !== JSON.stringify(expected)) {
    throw new Error(`Chapter numbers must be contiguous ${first}..${last}`);
  }
}
```

- [ ] **Step 4: Replace both public entry points**

Use a minimal document containing:

```html
<!doctype html><html lang="ru"><meta charset="utf-8">
<title>Математика под микроскопом</title>
<script>location.replace('kniga.html' + location.hash)</script>
<p><a id="fallback" href="kniga.html">Открыть полную книгу</a></p>
<script>document.getElementById('fallback').href='kniga.html'+location.hash</script>
</html>
```

- [ ] **Step 5: Run validation and tests**

Run:

```bash
node MathBook/scripts/validate-book.mjs --last-chapter 18
node --test MathBook/tests/book-structure.test.mjs
```

Expected: all checks PASS.

- [ ] **Step 6: Commit entry-point repair**

```bash
git add MathBook/frame.html MathBook/index.html MathBook/scripts MathBook/tests
git commit -m "fix: route MathBook entry points to complete book"
```

---

### Task 4: Extend the Map, Navigation, Styling, and Content Contracts

**Files:**
- Modify: `MathBook/src/shell.html`
- Create: `MathBook/tests/chapter-content.test.mjs`
- Modify: `MathBook/scripts/book-lib.mjs`

**Interfaces:**
- Produces CSS domain: `--violet:#6B4FA3`, `.lg-q`, and QLS route styling.
- Produces station labels for `ch-19` through `ch-25` in both `I18N.ru.stn` and `I18N.en.stn`.
- Produces: `validateNewChapter(pair): void`, enforcing the pedagogical contract.

- [ ] **Step 1: Write failing shell-contract tests**

```js
test('shell contains the complete QLS route', async () => {
  const shell = await readFile(new URL('../src/shell.html', import.meta.url), 'utf8');
  assert.match(shell, /--violet:#6B4FA3/);
  for (let n = 19; n <= 25; n += 1) {
    assert.match(shell, new RegExp(`data-go="ch-${n}"`));
    assert.match(shell, new RegExp(`'ch-${n}'`));
  }
  assert.match(shell, /QLS Extension/);
});
```

- [ ] **Step 2: Run the shell test**

Run: `node --test MathBook/tests/chapter-content.test.mjs`

Expected: FAIL because the QLS route is absent.

- [ ] **Step 3: Extend the course map without disturbing existing stations**

Add a violet route branching after the current chapter 18 station and place stations 19–25 in a second SVG row or continuation panel with a responsive minimum width. Add the legend label `QLS Extension`; do not compress the existing station spacing below its current value.

- [ ] **Step 4: Add bilingual station labels**

Use these exact short labels:

```js
// Russian
'ch-19':'Матрицы','ch-20':'PCA','ch-21':'Производные','ch-22':'ОДУ',
'ch-23':'Распределения','ch-24':'Вывод','ch-25':'Модели'
// English
'ch-19':'Matrices','ch-20':'PCA','ch-21':'Derivatives','ch-22':'ODEs',
'ch-23':'Distributions','ch-24':'Inference','ch-25':'Models'
```

- [ ] **Step 5: Implement the new-chapter validator**

For chapters 19–25, count `.example`, `.quiz`, `.problem`, `.boss`, `.cheat`, `.feynman`, and `<pre><code class="language-python">`. Require 3, 4, 6, 1, 1, 1, and 1 respectively in each language section. Also require at least one `.problem.flashback` or a heading/text containing `Найди ошибку` / `Find the error`.

- [ ] **Step 6: Run tests**

Run: `node --test MathBook/tests/*.test.mjs`

Expected: PASS for shell behavior; chapter tests have no new fragments yet and therefore only validate available files.

- [ ] **Step 7: Commit the QLS navigation contract**

```bash
git add MathBook/src/shell.html MathBook/scripts/book-lib.mjs MathBook/tests
git commit -m "feat: add QLS extension route to MathBook"
```

---

### Task 5: Author Chapter 19 — Vectors, Matrices, and Linear Systems

**Files:**
- Create: `MathBook/src/chapters/ch-19.html`
- Modify: `MathBook/tests/chapter-content.test.mjs`

**Interfaces:**
- Produces bilingual sections `#ch-19` and `#ch-19-en` with quiz names prefixed `q19-ru-` and `q19-en-`.

- [ ] **Step 1: Add a failing curriculum test**

```js
test('chapter 19 covers the required linear algebra bridge', async () => {
  const pair = await chapter(19);
  for (const term of ['dot product', 'matrix multiplication', 'Gaussian elimination', 'rank', 'NumPy']) {
    assert.match(pair.en, new RegExp(term, 'i'));
  }
  assert.match(pair.ru, /скалярн|матричн|метод Гаусса|ранг|NumPy/i);
  validateNewChapter(pair);
});
```

- [ ] **Step 2: Run the chapter test**

Run: `node --test MathBook/tests/chapter-content.test.mjs --test-name-pattern="chapter 19"`

Expected: FAIL because `ch-19.html` is missing.

- [ ] **Step 3: Write the bilingual chapter**

Use a gene-expression table as the opening hook. The worked examples must calculate: a vector norm and dot product; a valid `2×3 · 3×1` multiplication with explicit dimension checking; and a three-variable balance system solved by elimination. The deliberate-error task must reject element-wise multiplication as matrix multiplication. The boss solves a small nutrient-mixture model and verifies the solution by substitution.

The Python lab must contain executable NumPy code using `np.array`, `@`, `np.linalg.solve`, `np.linalg.matrix_rank`, and assertions for the displayed results.

- [ ] **Step 4: Validate the standalone source fragment**

Run:

```bash
node --test MathBook/tests/chapter-content.test.mjs --test-name-pattern="chapter 19"
```

Expected: PASS. Do not rebuild the public monolith until all advertised QLS map stations have matching chapter fragments in Task 12.

- [ ] **Step 5: Commit chapter 19**

```bash
git add MathBook/src/chapters/ch-19.html MathBook/tests
git commit -m "feat: teach matrices and linear systems in MathBook"
```

---

### Task 6: Author Chapter 20 — Eigenvectors and PCA

**Files:**
- Create: `MathBook/src/chapters/ch-20.html`
- Modify: `MathBook/tests/chapter-content.test.mjs`

**Interfaces:**
- Produces bilingual sections `#ch-20` and `#ch-20-en`, quiz prefix `q20-*`.

- [ ] **Step 1: Add and run a failing curriculum test**

Require both languages to cover transformations, basis, eigenvalues/eigenvectors, centering, standardization, covariance matrix, PCA scores/loadings, explained variance, and the unscaled-feature trap.

Run: `node --test MathBook/tests/chapter-content.test.mjs --test-name-pattern="chapter 20"`

Expected: FAIL because the file is absent.

- [ ] **Step 2: Write the bilingual chapter**

Open with a high-dimensional metabolomics dataset. Include a 2D transformation SVG, a hand-checkable eigenvector example using a diagonal or symmetric `2×2` matrix, a covariance example, and a PCA interpretation example. The boss compares PCA before and after scaling when one feature is measured in thousands and another in decimals.

The Python lab must use `StandardScaler` and `PCA(n_components=2)`, print `explained_variance_ratio_`, and label scores versus loadings accurately.

- [ ] **Step 3: Validate numerical claims and the standalone fragment**

Add a deterministic test for the displayed `2×2` eigenpair and explained-variance percentages. Run the chapter 20 test only; the public monolith remains at the last fully integrated version until Task 12.

- [ ] **Step 4: Commit chapter 20**

```bash
git add MathBook/src/chapters/ch-20.html MathBook/tests
git commit -m "feat: introduce eigenvectors and PCA"
```

---

### Task 7: Author Chapter 21 — Derivatives, Gradients, and Optimization

**Files:**
- Create: `MathBook/src/chapters/ch-21.html`
- Modify: `MathBook/tests/chapter-content.test.mjs`

**Interfaces:**
- Produces bilingual sections `#ch-21` and `#ch-21-en`, quiz prefix `q21-*`.

- [ ] **Step 1: Add and run a failing curriculum test**

Require limit intuition, derivative, tangent, power/exponential/logarithm/product/quotient/chain rules, partial derivative, gradient, critical point, loss function, gradient descent, and learning rate.

- [ ] **Step 2: Write the bilingual chapter**

Use changing drug concentration as the hook. Worked examples must differentiate a polynomial, `e^{-kt}`, and a composed dose-response expression; calculate a two-variable gradient; and minimize a quadratic loss. The deliberate-error task must expose a missing chain-rule factor. The boss performs several transparent gradient-descent steps and discusses learning rates that are too small or too large.

The Python lab uses NumPy to plot a loss curve and iteratively update one parameter with a fixed learning rate, recording convergence.

- [ ] **Step 3: Run the chapter 21 source tests**

Run: `node --test MathBook/tests/chapter-content.test.mjs --test-name-pattern="chapter 21"`

Expected: PASS; derivative values and the gradient-descent trace match deterministic assertions.

- [ ] **Step 4: Commit chapter 21**

```bash
git add MathBook/src/chapters/ch-21.html MathBook/tests
git commit -m "feat: teach calculus and optimization foundations"
```

---

### Task 8: Author Chapter 22 — Integrals, ODEs, and Numerical Methods

**Files:**
- Create: `MathBook/src/chapters/ch-22.html`
- Modify: `MathBook/tests/chapter-content.test.mjs`

**Interfaces:**
- Produces bilingual sections `#ch-22` and `#ch-22-en`, quiz prefix `q22-*`.

- [ ] **Step 1: Add and run a failing curriculum test**

Require antiderivative, definite integral, accumulated quantity, fundamental theorem intuition, separable ODE, exponential decay, logistic growth, equilibrium, Euler method, and step size.

- [ ] **Step 2: Write the bilingual chapter**

Use drug concentration and clearance as the hook. Worked examples must integrate a simple rate, solve `dC/dt=-kC`, identify logistic equilibria, and compute at least three Euler steps. Explicitly distinguish analytic and numerical solutions. The deliberate-error task must catch a dimensionally incorrect ODE or an Euler update with the missing step size. The boss compares Euler approximations at two step sizes against the analytic decay curve.

The Python lab implements Euler's method as a small function and compares it with `scipy.integrate.solve_ivp` and the analytic result.

- [ ] **Step 3: Run the chapter 22 source tests**

Run: `node --test MathBook/tests/chapter-content.test.mjs --test-name-pattern="chapter 22"`

Expected: PASS; Euler values and analytic values match stored tolerances.

- [ ] **Step 4: Commit chapter 22**

```bash
git add MathBook/src/chapters/ch-22.html MathBook/tests
git commit -m "feat: add integrals ODEs and numerical methods"
```

---

### Task 9: Author Chapter 23 — Random Variables and Distributions

**Files:**
- Create: `MathBook/src/chapters/ch-23.html`
- Modify: `MathBook/tests/chapter-content.test.mjs`

**Interfaces:**
- Produces bilingual sections `#ch-23` and `#ch-23-en`, quiz prefix `q23-*`.

- [ ] **Step 1: Add and run a failing curriculum test**

Require random variable, PMF, PDF, CDF, expectation, variance, Bernoulli, Binomial, Poisson, Normal, z-score, and Central Limit Theorem.

- [ ] **Step 2: Write the bilingual chapter**

Use sequencing-count noise as the hook. Worked examples must compute a discrete expectation and variance; a binomial adverse-event probability; a Poisson event probability; and a Normal z-score. The deliberate-error exercise must catch interpreting a density value as a probability. The boss selects and justifies appropriate distributions for four biological scenarios.

The Python lab simulates each distribution using `numpy.random.default_rng(seed=QLS_SEED)` where `QLS_SEED = 2026`, compares empirical and theoretical moments, and shows the sample-mean distribution narrowing.

- [ ] **Step 3: Run the chapter 23 source tests**

Run: `node --test MathBook/tests/chapter-content.test.mjs --test-name-pattern="chapter 23"`

Expected: PASS; exact discrete calculations pass and simulation assertions use fixed tolerances and seed 2026.

- [ ] **Step 4: Commit chapter 23**

```bash
git add MathBook/src/chapters/ch-23.html MathBook/tests
git commit -m "feat: teach probability distributions"
```

---

### Task 10: Author Chapter 24 — Statistical Inference, Likelihood, and Bayes

**Files:**
- Create: `MathBook/src/chapters/ch-24.html`
- Modify: `MathBook/tests/chapter-content.test.mjs`

**Interfaces:**
- Produces bilingual sections `#ch-24` and `#ch-24-en`, quiz prefix `q24-*`.

- [ ] **Step 1: Add and run a failing curriculum test**

Require estimator, bias, standard error, confidence interval, null/alternative hypotheses, p-value, Type I/II errors, power, multiple testing, effect size, likelihood, maximum likelihood, prior, posterior, and posterior predictive.

- [ ] **Step 2: Write the bilingual chapter**

Open with two studies reporting the same p-value but different effect sizes. Include exact warnings that a p-value is not `P(H0 is true | data)` and a 95% frequentist confidence interval is not a 95% posterior probability statement. Reuse the screening scenario from chapter 13 to derive Bayes by natural frequencies and then by formula. The deliberate-error exercise must repair a press-release interpretation of statistical significance. The boss reviews a small biomarker study for power, multiplicity, practical effect, and warranted claims.

The Python lab computes a bootstrap confidence interval with seed 2026, illustrates a likelihood curve for a binomial parameter, and performs a transparent Beta-Binomial update without hiding the prior.

- [ ] **Step 3: Run the chapter 24 source tests**

Run: `node --test MathBook/tests/chapter-content.test.mjs --test-name-pattern="chapter 24"`

Expected: PASS; the screening posterior, interval endpoints, and Beta posterior parameters pass numerical assertions.

- [ ] **Step 4: Commit chapter 24**

```bash
git add MathBook/src/chapters/ch-24.html MathBook/tests
git commit -m "feat: add inference likelihood and Bayes"
```

---

### Task 11: Author Chapter 25 — Linear Models and ML Mathematics

**Files:**
- Create: `MathBook/src/chapters/ch-25.html`
- Modify: `MathBook/tests/chapter-content.test.mjs`

**Interfaces:**
- Produces bilingual sections `#ch-25` and `#ch-25-en`, quiz prefix `q25-*`.

- [ ] **Step 1: Add and run a failing curriculum test**

Require design matrix, coefficient, prediction, residual, mean squared error, multiple regression, categorical feature, sigmoid, logistic regression, train/validation/test, regularization, and feature scaling.

- [ ] **Step 2: Write the bilingual chapter**

Use prediction of a clinical measurement as the hook. Worked examples must express predictions as `Xβ`, calculate residuals and MSE, interpret a coefficient with units, and convert a logit through the sigmoid. The deliberate-error exercise must identify data leakage caused by scaling before the split. The boss compares two biomedical models using validation behavior, effect interpretation, calibration awareness, and non-causal language.

The Python lab constructs a small DataFrame, performs a train/test split, fits a pipeline containing `StandardScaler` and either `LinearRegression` or `LogisticRegression`, reports an appropriate metric, and explains why prediction quality alone does not establish causality.

- [ ] **Step 3: Add a cumulative bridge diagram**

The final recap must explicitly connect:

```text
data table → matrix X → scaling → PCA/features → model → loss → gradient → fitted parameters → validated prediction
```

- [ ] **Step 4: Run the chapter 25 source tests**

Run: `node --test MathBook/tests/chapter-content.test.mjs --test-name-pattern="chapter 25"`

Expected: PASS. Task 12 performs the first complete 26-chapter build and cross-chapter validation.

- [ ] **Step 5: Commit chapter 25**

```bash
git add MathBook/src/chapters/ch-25.html MathBook/tests
git commit -m "feat: complete QLS mathematics bridge"
```

---

### Task 12: Validate Python Examples and the Complete Publication

**Files:**
- Create: `MathBook/scripts/validate-python-examples.py`
- Create: `MathBook/tests/browser-smoke.mjs`
- Modify: `MathBook/scripts/validate-book.mjs`

**Interfaces:**
- Consumes `<pre><code class="language-python">` blocks from chapter fragments.
- Produces a nonzero exit code for syntax errors, missing fixed seeds in stochastic labs, or duplicate code-block IDs.

- [ ] **Step 1: Write a failing syntax-extraction test**

```python
def extract_python_blocks(html: str) -> list[str]:
    pattern = r'<pre><code class="language-python"[^>]*>(.*?)</code></pre>'
    return [html_unescape(block) for block in re.findall(pattern, html, flags=re.S)]

def validate_syntax(block: str, label: str) -> None:
    compile(block, label, 'exec')
```

Create a fixture containing one valid and one invalid block and assert that the invalid block returns a nonzero CLI status.

- [ ] **Step 2: Implement syntax and reproducibility checks**

Reject stochastic blocks containing `random` or `default_rng` unless they also contain `2026` or an explicit deterministic seed variable assigned to 2026. HTML-unescape `&lt;`, `&gt;`, `&amp;`, and quotes before compilation.

- [ ] **Step 3: Add smoke tests for generated router/state contracts**

Assert that the generated file retains `mikroskop-v1`, uses all `.chapter` nodes to build order, contains `ch-18 → ch-19 → ... → ch-25`, contains 26 map stations, and has no unfilled `<!--CHAPTERS-->` marker.

- [ ] **Step 4: Run the full automated verification**

Run:

```bash
node MathBook/scripts/build-book.mjs
node MathBook/scripts/validate-book.mjs --last-chapter 25
node --test MathBook/tests/*.test.mjs
python3 MathBook/scripts/validate-python-examples.py
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 5: Commit complete validation**

```bash
git add MathBook/scripts MathBook/tests MathBook/kniga.html
git commit -m "test: validate complete QLS MathBook"
```

---

### Task 13: Document, Browser-Test, and Publish

**Files:**
- Create: `MathBook/README.md`
- Modify: root `README.md` only if it currently links directly to `frame.html`; otherwise leave it unchanged.

**Interfaces:**
- Documents the canonical public URL: `MathBook/kniga.html#hub`.
- Documents compatibility URLs: `MathBook/index.html` and `MathBook/frame.html`.

- [ ] **Step 1: Write editing and build documentation**

Document exact commands:

```bash
node MathBook/scripts/build-book.mjs
node MathBook/scripts/validate-book.mjs --last-chapter 25
node --test MathBook/tests/*.test.mjs
python3 MathBook/scripts/validate-python-examples.py
```

Explain that `kniga.html` is generated and committed, chapter sources are authoritative, and existing chapters must not be edited in the generated file.

- [ ] **Step 2: Serve the repository locally**

Run: `python3 -m http.server 8765 --directory .`

Expected: server reports `Serving HTTP on ... port 8765`.

- [ ] **Step 3: Verify desktop behavior in the browser**

At `http://localhost:8765/MathBook/frame.html#ch-19`, confirm redirection to `kniga.html#ch-19`. Verify map, table of contents, rail, previous/next navigation from chapters 18 through 25, Russian/English switching, quizzes, hints, solutions, and completion stamps. Reload and confirm existing progress persists.

- [ ] **Step 4: Verify mobile and print behavior**

At a 390×844 viewport, confirm bottom navigation and no horizontal overflow in chapter prose, tables, figures, or code blocks. Open print preview and confirm navigation controls are absent and only the selected language is printed.

- [ ] **Step 5: Run final regression commands**

```bash
node MathBook/scripts/build-book.mjs
node MathBook/scripts/validate-book.mjs --last-chapter 25
node --test MathBook/tests/*.test.mjs
python3 MathBook/scripts/validate-python-examples.py
git diff --check
git status --short
```

Expected: tests PASS; only intentional README/generated publication changes remain before the documentation commit.

- [ ] **Step 6: Commit documentation and verified publication**

```bash
git add MathBook/README.md MathBook/kniga.html MathBook/frame.html MathBook/index.html README.md
git commit -m "docs: publish QLS MathBook extension"
```

- [ ] **Step 7: Push and verify GitHub Pages**

Push the implementation branch, merge through the user's chosen workflow, wait for the existing Pages deployment, and verify:

- `/NeuroHand/MathBook/kniga.html#hub` shows 26 experiments;
- `/NeuroHand/MathBook/frame.html#ch-25` redirects to the final chapter;
- the course map and language switch work on the deployed origin.

Do not claim publication success until the deployed URLs have been opened and checked.
