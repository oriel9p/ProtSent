# Mechanical compliance check — FINAL_rebuttal.md

Checked file: `/home/ddofer/ProtSent/rebuttal/FINAL_rebuttal.md` (475 lines, 28,321 bytes).
All counts measured, not estimated.

## 1. Character counts — ALL THREE PASS

Method (reproducible):

```
python3 -c "
import re
src=open('/home/ddofer/ProtSent/rebuttal/FINAL_rebuttal.md',encoding='utf-8').read()
for r in ['HNXd','jVGf','Yi1G']:
    b=re.search(r'<!-- BEGIN %s -->\n(.*?)\n<!-- END %s -->'%(r,r),src,re.S).group(1).strip()
    print(r,len(b),len(b.encode()))"
```

Paste unit = everything between `<!-- BEGIN X -->` and `<!-- END X -->`, exclusive,
`.strip()`ed. Excludes the document title, the naming preamble (lines 1-9), the
`## Response to Reviewer X` headings, the `---` separators and the marker comments —
none of those get pasted into a reply box.

| response | characters (paste unit) | UTF-8 bytes | margin to 10,000 | verdict |
|---|---|---|---|---|
| HNXd | **9,874** | 9,911 | 126 | under |
| jVGf | **7,821** | 7,846 | 2,179 | under |
| Yi1G | **9,943** | 9,976 | 57 | under |

Character count is the right unit: OpenReview limits string length, and the file
contains no astral-plane characters (verified — only U+2014 em dash x46, U+00B1 x5,
U+2192 x1), so Python `len()` == JavaScript `.length`.

**Nothing is at or over 10,000.** The brief said Yi1G was ~10,009 and HNXd ~9,946;
neither reproduces against the current file. The closest reproduction of 10,009 is
Yi1G's UTF-8 byte count *with* the `## Response to Reviewer Yi1G` heading included
= 10,006, i.e. the earlier count measured bytes and included the heading. On the
paste unit Yi1G is 9,943.

Margins are thin: Yi1G has 57 characters of headroom, HNXd 126. Any edit to those
two must be length-neutral or shrinking. jVGf has 2,179 characters spare.

## 2. Links, and references to things the reviewer cannot see

**URLs / link syntax: none.** Zero matches for `http://`, `https://`, `www.`,
`](`, `[[`, `<a`, `![`, `.com`, `.org`, `.io`, `github`, `huggingface`, `zenodo`,
`anonymous`, `attach`, `supplementary`, `see the`, `as shown`. The only
square-bracket text outside numeric confidence intervals is line 397
`np.concatenate([emb[s1], emb[s2]])`, inside a code span — not link syntax.

**Zero occurrences of "revised", "revision", "manuscript", "Appendix", "Figure",
"Fig.", "Section", "Sec."** anywhere in the file.

Three table references, all to the **submitted** paper, all verified against
`PAPER_text.txt` — all legitimate, reviewers hold that PDF:

| line | text | judgement |
|---|---|---|
| 142 | `### 4-5. Seed variability and Table 5 (your questions 4 and 5)` | LEGITIMATE. Paper line 369: "Table 5: Few-shot evaluation (ESM-2 35M, KNN probe). Relative improvement". HNXd's own review asks about Table 5 twice (REVIEWS_actual.md lines 26, 37). |
| 144 | "So Table 5 is withdrawn rather than re-presented in absolute units" | LEGITIMATE, same table, and it answers the reviewer's literal request. |
| 388 | "(35M: per-device 64 x 16 steps; 150M: 16 x 64, our Table 6)" | LEGITIMATE and numerically correct. Paper line 558 Table 6: per-device 64/16, grad accumulation 16/64, effective 1024. Note the paper puts Table 6 in Appendix 8 but the rebuttal says only "our Table 6", so no appendix reference is made. |
| 468 | "do not match our Table 1 (32.9M / 133.9M / 36.5M)" | LEGITIMATE and correct. Paper lines 140-143: Pfam 32.9M, AFDB50 133.9M, STRING 36.5M. |

No reference anywhere to a revised table, a revision, or the repo.

## 3. Placeholders

**None.** Zero matches for `[[RESULT]]`, `RESULT`, `TODO`, `TBD`, `XXX`, `FIXME`,
`PLACEHOLDER`. Every bracketed span is a confidence interval.

## 4. Markdown that renders badly

- **HTML: only the six `<!-- BEGIN/END X -->` marker comments**, which are outside the
  paste unit by construction. No `<a>`, `<img>`, `<br>`, `<div>`, `<sup>`, `<sub>`,
  `<table>`.
- **Images: none.**
- **Nested tables: none.** Five tables total, each a contiguous block of `|` lines:

| lines | rows | columns | separator row valid |
|---|---|---|---|
| 24-27 | 4 | 3 (uniform) | yes |
| 57-62 | 6 | 5 (uniform) | yes |
| 110-115 | 6 | 5 (uniform) | yes |
| 117-122 | 6 | 4 (uniform) | yes |
| 213-218 | 6 | 5 (uniform) | yes |

  Every table row is a single unwrapped line, column counts are uniform within each
  table, and the two adjacent tables at 110-115 and 117-122 are separated by a blank
  line (116) so they render as two tables, not one broken one.
- **No accidental lists or blockquotes.** No line starts with `- `, `+ `, `* `, `> `,
  `N. `, or 4+ spaces, so no hard-wrapped continuation line (e.g. one starting with a
  `+0.0289` interval) will be swallowed into a list.
- Underscored identifiers (`remote_homology`, `HLA_pseudoseq|peptide`, `fold_prediction`
  at line 328) are either backticked or intra-word; CommonMark/markdown-it does not
  emphasise intra-word underscores, so no accidental italics. The one bare pipe
  (`HLA_pseudoseq|peptide`, lines 174 and 398) is inside a code span and is not in a
  table row, so it cannot split a cell.
- Hard-wrapped paragraphs at ~85 chars are cosmetic only; they reflow (or become soft
  breaks) either way.

## 5. Are tables unambiguous read cold?

Three are clean, two need a header word. None is a compliance failure.

**CLEAN — line 57-62 (HNXd).** Header `| method (all 2,207 queries) |` names the
denominator; lead-in names dataset (SCOPe-40), level (family), gallery size, self
excluded, no-hit = failure, and the 0.7671 ceiling; rows name model and size
("ESM-2 35M", "ProtSent-V1 35M (submitted)", "ProtSent-V2 35M (retrained)"); columns
name the metric (R@1/R@10/R@30/MAP).

**CLEAN — line 110-115 (HNXd).** Header `| method (1,693 eligible queries) |` makes the
different denominator explicit, and the sentence above says so too.

**CLEAN — line 117-122 (HNXd).** Header `| paired difference |` plus explicit
`Recall@1 / Recall@10 / MAP` columns; the paragraph immediately above fixes the
denominator (1,693 eligible) and the procedure (10,000 paired resamples). Sign
convention is inferable from the row labels (`V2 - MMseqs2`) but is never stated;
the surrounding prose switches orientation words ("MMseqs2 beats ProtSent-V1 at
top-1 by +0.0697" vs "MMseqs2 vs ESM-2 at R@10 (-0.0213)"), so a cold reader has
to derive that both prose deltas are MMseqs2 - ESM-2. Minor.

**NEEDS ONE WORD — line 24-27 (HNXd).** `| probe | ProtSent-V1 | ProtSent-V2 |` with
cells "11 win / 3 tie / 6 lose, median +0.0075". Two things live only in surrounding
prose: (a) the comparison is *against ESM-2 35M* (stated in the lead-in sentence, one
line above — acceptable); (b) the lead-in says "all 23 tasks" but the counts sum to
**20**, and that is only reconciled 4 lines later in note (ii). Suggest header
`| probe (20 comparable tasks, test split, vs ESM-2 35M) | ... |`, which also costs
nothing net if the redundant part of the lead-in shortens. Also "median" is a median
delta on each task's main metric — only clarified by note (i)'s "±0.005 on each task's
main metric".

**NEEDS ONE WORD — line 213-218 (jVGf).** Header is bare `| method |` where HNXd's
identical table says `| method (all 2,207 queries) |`. The lead-in sentence mentions
both 2,207 and 1,693 in the same breath, so a cold reader can attach the numbers to
the wrong denominator. jVGf has 2,179 characters of margin — add
`(all 2,207 queries)` there. Row labels and metric columns are otherwise fine.

**Yi1G has no tables**; its retrieval numbers are inline and do carry metric, model
and denominator ("On SCOPe-40 (family level, 2,207-sequence gallery ... only 1,693
queries have a non-self same-family neighbour), R@1 / R@10 / MAP: MMseqs2 0.5029 /
0.5637 / 0.3100; ESM-2 35M ...; V1 ...; V2 ..."). One inconsistency: in section 8
"MMseqs2 beats ESM-2 at top-1 by +0.1565" is the only delta in that paragraph quoted
without its interval, while every neighbouring delta has one.

## Summary

| item | result |
|---|---|
| 1. length | PASS — HNXd 9,874 / jVGf 7,821 / Yi1G 9,943 characters. None >= 10,000. Yi1G margin 57. |
| 2. links & unseeable references | PASS — no URLs, no link syntax; only Tables 1, 5, 6 of the submitted paper, all verified correct. |
| 3. placeholders | PASS — none. |
| 4. markdown | PASS — 5 well-formed tables, no HTML in the paste unit, no images, no accidental lists. |
| 5. table clarity | PASS with 2 minor edits: add `(all 2,207 queries)` to the jVGf table header (line 213); add task count / comparison target to the HNXd probe-table header (line 24). |
