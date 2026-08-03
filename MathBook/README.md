# MathBook

Self-contained bilingual mathematics book for personal study. Open `frame.html`,
`index.html`, or `kniga.html` in a browser. The compatibility entry points forward
to the complete publication and preserve chapter hashes such as `#ch-19`.

The book now contains 26 Russian/English chapter pairs:

- chapters 0–18: the original mathematics course;
- chapters 19–25: the QLS bridge covering linear algebra, PCA, calculus,
  differential equations, probability distributions, statistical inference,
  and linear models.

The QLS chapters include biological/medical examples, short self-checks, and
small Python/NumPy experiments.

## Rebuild after editing

```bash
node MathBook/scripts/build-book.mjs
node MathBook/scripts/validate-book.mjs --last-chapter 25
```

The editable sources are in `src/shell.html` and `src/chapters/`. The generated,
single-file publication is `kniga.html`.
