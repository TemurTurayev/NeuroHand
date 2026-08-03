# MathBook QLS Extension — Design Specification

## 1. Objective

Extend *Mathematics Under the Microscope* from a Digital SAT mathematics course into a continuous bridge toward Constructor University's M.Sc. Quantitative Life Science curriculum.

The existing chapters 0–18 remain intact as the prerequisite foundation. A new, clearly separated `QLS Extension` route adds seven university-level chapters, numbered 19–25. The published book remains bilingual, interactive, mobile-friendly, usable from GitHub Pages, and available as a single self-contained HTML document.

Success means that a learner who completes the extension has a practical first-pass foundation for:

- the mathematics portion of Guided Self-Study;
- Data Tools for the Life Sciences;
- Computational Life Science;
- introductory quantitative work in laboratory rotations;
- later study of machine learning and biological data analysis.

The extension is a bridge, not a substitute for university lectures or a rigorous mathematics degree.

## 2. Scope

### Included

- Seven new bilingual chapters with the same pedagogical structure and visual language as chapters 0–18.
- Medical, biological, omics, neuroscience, and pharmacological examples.
- Static Python/Jupyter examples that can be copied and run locally.
- Interactive quizzes, progressive hints, worked solutions, chapter bosses, cheat sheets, and Feynman tasks.
- A new QLS branch on the course map and navigation rail.
- Modular source files and a deterministic build process that generates the standalone published book.
- Repair of the currently incomplete `frame.html` entry point while preserving existing URLs and hash links.
- Automated structural validation and proportionate browser-based verification.

### Excluded

- Formal real analysis, proof-heavy linear algebra, measure-theoretic probability, advanced numerical analysis, and full mathematical statistics.
- An in-browser Python runtime, package installation, accounts, cloud synchronization, or a backend.
- Rewriting or materially changing the educational content of chapters 0–18.
- Turning the MathBook into a replacement for the separate QLS Interactive Book.

## 3. Curriculum

### Chapter 19 — Vectors, Matrices, and Linear Systems

Core topics:

- scalars, vectors, and matrices;
- vector addition, scalar multiplication, dot product, norms, and distance;
- matrix shapes, indexing, transpose, and matrix multiplication;
- identity and inverse matrices, with emphasis on when an inverse does not exist;
- systems in matrix form `Ax = b`;
- Gaussian elimination, rank, and linear independence at an intuitive level;
- NumPy arrays and basic matrix operations.

Primary context: mixtures, gene-expression tables, and solving small physiological balance models.

### Chapter 20 — Transformations, Eigenvectors, and PCA

Core topics:

- matrices as transformations;
- basis and change of coordinates;
- eigenvectors and eigenvalues through invariant directions;
- covariance matrices;
- centering and standardization;
- PCA intuition, explained through eigenvectors and variance;
- interpreting scores, loadings, and explained variance;
- limitations of PCA and the danger of unscaled features.

Primary context: reducing an omics dataset and visualizing patient groups.

### Chapter 21 — Derivatives, Gradients, and Optimization

Core topics:

- limits as preparation for derivatives, without proof-heavy treatment;
- derivative as local rate of change and tangent slope;
- power, exponential, logarithmic, product, quotient, and chain rules;
- partial derivatives and gradients;
- critical points, minima, and maxima;
- gradient descent and learning rate;
- loss functions and overfitting intuition.

Primary context: concentration curves, dose-response relationships, and minimizing model error.

### Chapter 22 — Integrals, Differential Equations, and Numerical Methods

Core topics:

- antiderivatives and definite integrals;
- integral as accumulated quantity and area;
- the fundamental theorem of calculus at an intuitive level;
- separable first-order ODEs;
- exponential growth and decay;
- logistic growth;
- one- and two-compartment modeling intuition;
- equilibrium and stability at an introductory level;
- Euler's method and the reason step size matters;
- numerical solutions in Python.

Primary context: pharmacokinetics, bacterial growth, clearance, and population models.

### Chapter 23 — Random Variables and Probability Distributions

Core topics:

- discrete and continuous random variables;
- probability mass and density functions;
- cumulative distribution functions;
- expectation, variance, and standard deviation;
- Bernoulli, Binomial, Poisson, and Normal distributions;
- z-scores and standardization;
- Central Limit Theorem intuition;
- simulation and sampling with NumPy.

Primary context: adverse events, sequencing counts, mutation counts, and laboratory measurement noise.

### Chapter 24 — Statistical Inference, Likelihood, and Bayes

Core topics:

- populations, samples, estimators, bias, and standard error;
- confidence intervals and their correct interpretation;
- null and alternative hypotheses;
- p-values, Type I and Type II errors, power, and multiple-testing intuition;
- effect size versus statistical significance;
- likelihood and maximum-likelihood estimation;
- prior, likelihood, posterior, and posterior predictive intuition;
- Bayes' theorem connected to the existing screening example;
- reproducible reporting and common scientific interpretation errors.

Primary context: clinical trials, diagnostic tests, biomarkers, and high-dimensional biological studies.

### Chapter 25 — Linear Models and the Mathematics of Machine Learning

Core topics:

- simple and multiple linear regression;
- design matrix, coefficients, predictions, residuals, and mean squared error;
- categorical variables at an introductory level;
- classification and the sigmoid/logistic-regression idea;
- train, validation, and test separation;
- regularization intuition;
- feature scaling;
- connection among matrix multiplication, gradients, optimization, PCA, and model fitting;
- a small end-to-end biomedical modeling example in Python.

Primary context: predicting a clinical measurement and classifying a biomedical sample without making causal claims.

## 4. Pedagogical Template

Every new chapter follows the existing book's recognizable sequence:

1. A narrative hook grounded in medicine, biology, or data science.
2. `Try it first`, designed to expose the learner's current intuition.
3. Conceptual explanation before formal notation.
4. Definitions and formulas with symbol legends.
5. At least three fully worked examples.
6. Visual explanation using inline SVG, a table, or an existing graph-paper component.
7. At least four interactive hypothesis checks.
8. At least six exercises distributed across levels A, B, and C, each with progressive hints and a complete solution.
9. At least one deliberate-error or flashback exercise.
10. A biomedical chapter boss combining multiple concepts.
11. A runnable Python/Jupyter mini-lab using common libraries such as NumPy, pandas, matplotlib, scipy, or scikit-learn when appropriate.
12. `Now you can`, a concise cheat sheet, and a Feynman explanation task.

Russian and English versions must teach the same concepts and use equivalent examples and answers. Terminology should include the standard English term where that helps future English-language study.

## 5. Information Architecture and Visual Design

The current map receives a fifth route named `QLS Extension`, visually distinct from Algebra, Advanced Mathematics, Data Analysis, and Geometry/Trigonometry. The route begins after chapter 18 and contains stations 19–25.

The new route uses a restrained violet accent that remains legible against the existing paper, ink, and grid palette. It must not change the established laboratory-journal aesthetic.

Navigation requirements:

- chapters 19–25 appear in the desktop rail, mobile navigation, course map, and generated table of contents;
- previous/next navigation crosses correctly from chapter 18 to chapter 19;
- existing hashes such as `#ch-12` continue to work;
- new hashes use `#ch-19` through `#ch-25`;
- progress from the existing local-storage namespace is preserved;
- the completion denominator expands from 19 to 26 without deleting existing completion and quiz state.

## 6. Source and Build Architecture

The present `MathBook/kniga.html` is a 2.27 MB generated-style monolith, while `MathBook/frame.html` contains the shell but no chapter sections. The extension should not continue manual editing of a single giant file.

Proposed structure:

```text
MathBook/
├── src/
│   ├── shell.html
│   └── chapters/
│       ├── ch-00.html
│       ├── ...
│       ├── ch-18.html
│       ├── ch-19.html
│       └── ... ch-25.html
├── scripts/
│   ├── split-existing-book.mjs
│   ├── build-book.mjs
│   └── validate-book.mjs
├── frame.html
├── index.html
└── kniga.html
```

`split-existing-book.mjs` is a one-time, deterministic migration tool that extracts the existing shell and bilingual chapter pairs from `kniga.html` without rewriting their content. Its output is reviewed before extension content is added.

`build-book.mjs` concatenates the shell and ordered chapter fragments into a standalone `kniga.html`. The generated file is committed so GitHub Pages requires no server-side build step.

`frame.html` becomes a lightweight compatibility redirect to `kniga.html` while preserving `location.hash`. `index.html` also points to the complete book. Therefore the user's existing public link continues to work rather than opening a zero-chapter shell.

Each chapter fragment contains the Russian and English section nodes required by the current router. IDs, `data-ch`, quiz radio names, and answer keys must be unique and deterministic.

Python examples are rendered as static, copyable code blocks. They do not execute arbitrary code in the browser and do not introduce a Pyodide-sized dependency.

## 7. Validation and Testing

### Automated validation

The validation script must fail the build when it finds:

- a missing chapter number from 0 through 25;
- duplicate section IDs, quiz names, or form IDs;
- missing Russian/English chapter parity;
- malformed hash targets or navigation order;
- missing headings, chapter bosses, cheat sheets, or Feynman tasks in new chapters;
- fewer than the required quizzes or exercises;
- accidental external runtime dependencies;
- an empty `<!--CHAPTERS-->` slot in the generated publication.

Where formulas have numerical worked answers, small deterministic checks should verify the values used in examples and solutions.

### Browser verification

Verify at desktop and mobile viewport sizes:

- `index.html`, `frame.html`, and `kniga.html` reach the complete book;
- map, rail, table of contents, previous/next controls, and direct hashes work;
- language switching shows the matching chapter;
- quiz feedback, hints, solutions, and completion stamps work;
- existing progress survives the expanded chapter count;
- chapters remain readable without horizontal overflow;
- printing does not include navigation controls and includes both the intended content and formulas cleanly.

## 8. Content Quality Requirements

- Explanations prioritize intuition but must not replace correct definitions with misleading metaphors.
- Medical examples must distinguish educational simplification from clinical guidance.
- Statistical language must avoid common errors such as treating a p-value as the probability that the null hypothesis is true or treating confidence intervals as posterior probabilities.
- ML examples must separate prediction, association, and causation.
- Units, matrix dimensions, probability ranges, and numerical results must be checked consistently.
- Each Python mini-lab must be syntactically valid and reproducible with a stated random seed when randomness is used.

## 9. Delivery Criteria

The extension is complete when:

- all seven bilingual chapters are present and pass structural validation;
- the existing nineteen chapters remain content-equivalent after modularization;
- a clean build reproducibly generates `kniga.html`;
- the old `frame.html#...` URLs reach the correct complete chapter;
- browser verification passes on desktop and mobile;
- the generated book is published through the repository's existing GitHub Pages workflow;
- no user progress is intentionally discarded.

## 10. Implementation Sequence

1. Protect the current publication with snapshots and structural counts.
2. Extract the monolith into modular sources and prove a content-equivalent rebuild.
3. Repair the public entry points.
4. Extend navigation, map, styling, and validation for chapters 19–25.
5. Author and verify chapters in prerequisite order.
6. Build the complete book, run automated validation, and perform browser checks.
7. Commit the generated publication and deploy through GitHub Pages.
