import { createHash } from 'node:crypto';

export function sha256(text) {
  return createHash('sha256').update(text).digest('hex');
}

export function chapterSections(html) {
  const starts = [
    ...html.matchAll(
      /<section class="chapter" id="ch-(\d+)(-en)?"[^>]*>/g,
    ),
  ];

  return starts.map((match, index) => {
    const nextStart = starts[index + 1]?.index;
    const end = nextStart ?? html.indexOf('</main>', match.index);
    if (end < 0) {
      throw new Error(`Cannot find the end of chapter ${match[1]}`);
    }
    return {
      number: Number(match[1]),
      lang: match[2] ? 'en' : 'ru',
      html: html.slice(match.index, end),
    };
  });
}

export function extractChapterPairs(html) {
  const grouped = new Map();
  for (const section of chapterSections(html)) {
    const pair = grouped.get(section.number) ?? { number: section.number };
    if (pair[section.lang]) {
      throw new Error(
        `Duplicate ${section.lang} section for chapter ${section.number}`,
      );
    }
    pair[section.lang] = section.html;
    grouped.set(section.number, pair);
  }

  return [...grouped.values()]
    .sort((left, right) => left.number - right.number)
    .map((pair) => {
      if (!pair.ru || !pair.en) {
        throw new Error(`Missing bilingual twin for chapter ${pair.number}`);
      }
      return { ...pair, combined: pair.ru + pair.en };
    });
}

export function splitBook(sourceHtml) {
  const sections = chapterSections(sourceHtml);
  if (sections.length === 0) {
    throw new Error('Book contains no chapter sections');
  }
  const firstStart = sourceHtml.indexOf(sections[0].html);
  const mainEnd = sourceHtml.indexOf('</main>', firstStart);
  if (firstStart < 0 || mainEnd < 0) {
    throw new Error('Cannot isolate chapter range from book shell');
  }
  const chapters = new Map(
    extractChapterPairs(sourceHtml).map((pair) => [pair.number, pair.combined]),
  );
  return {
    shell:
      sourceHtml.slice(0, firstStart) +
      '<!--CHAPTERS-->' +
      sourceHtml.slice(mainEnd),
    chapters,
  };
}

export function buildBook(shell, chapterFiles) {
  const marker = '<!--CHAPTERS-->';
  if ((shell.match(/<!--CHAPTERS-->/g) ?? []).length !== 1) {
    throw new Error('Shell must contain exactly one CHAPTERS marker');
  }
  const body = [...chapterFiles]
    .sort((left, right) => left.number - right.number)
    .map((chapter) => chapter.html)
    .join('');
  return shell.replace(marker, body);
}

export function assertContiguousNumbers(numbers, first, last) {
  const expected = Array.from(
    { length: last - first + 1 },
    (_, index) => first + index,
  );
  if (JSON.stringify(numbers) !== JSON.stringify(expected)) {
    throw new Error(`Chapter numbers must be contiguous ${first}..${last}`);
  }
}

function uniqueAttributeValues(html, attribute) {
  return new Set(
    [...html.matchAll(new RegExp(`${attribute}="([^"]+)"`, 'g'))].map(
      (match) => match[1],
    ),
  );
}

export function validateBook(html, expectedLastChapter) {
  const pairs = extractChapterPairs(html);
  assertContiguousNumbers(
    pairs.map((pair) => pair.number),
    0,
    expectedLastChapter,
  );

  const ids = [...html.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]);
  const duplicateIds = ids.filter((id, index) => ids.indexOf(id) !== index);
  if (duplicateIds.length > 0) {
    throw new Error(`Duplicate element id: ${duplicateIds[0]}`);
  }

  const quizOwners = new Map();
  for (const pair of pairs) {
    for (const lang of ['ru', 'en']) {
      for (const name of uniqueAttributeValues(pair[lang], 'name')) {
        const owner = `ch-${pair.number}-${lang}`;
        if (quizOwners.has(name)) {
          throw new Error(
            `Duplicate quiz radio name ${name} in ${quizOwners.get(name)} and ${owner}`,
          );
        }
        quizOwners.set(name, owner);
      }
    }
  }

  if (/<script\b[^>]*\bsrc=|<link\b[^>]*rel="stylesheet"/i.test(html)) {
    throw new Error('Publication must not depend on external runtime assets');
  }
  if (html.includes('<!--CHAPTERS-->')) {
    throw new Error('Generated publication contains an unfilled CHAPTERS marker');
  }

  const knownChapters = new Set(pairs.map((pair) => `ch-${pair.number}`));
  for (const match of html.matchAll(/href="#(ch-\d+)"/g)) {
    if (!knownChapters.has(match[1])) {
      throw new Error(`Broken internal chapter link: #${match[1]}`);
    }
  }

  return { chapterPairs: pairs.length, quizNames: quizOwners.size };
}
