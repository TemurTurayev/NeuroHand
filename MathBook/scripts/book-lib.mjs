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
