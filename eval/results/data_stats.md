# Data statistics

Edits in the edit table built from the BEA-2019 releases of W&I+LOCNESS and FCE (gold ERRANT edits, annotator 0, `noop` and `UNK` edits excluded). FCE essays have no individual CEFR rating and are counted as B2, the level of the FCE exam. Essays with no edits are not counted.

## Essays and edits per corpus and split

| Corpus / split | Essays | Edits |
| --- | ---: | ---: |
| fce / dev | 159 | 3421 |
| fce / test | 194 | 4549 |
| fce / train | 2114 | 43051 |
| locness / dev | 50 | 929 |
| wi / dev | 299 | 6532 |
| wi / train | 2996 | 62034 |
| total | 5812 | 120516 |

## Edits per CEFR level

| Level | train | dev | test | total |
| --- | ---: | ---: | ---: | ---: |
| A1 | 8506 | 952 | 0 | 9458 |
| A2 | 19679 | 1996 | 0 | 21675 |
| B1 | 15090 | 1478 | 0 | 16568 |
| B2 | 51823 | 4441 | 4549 | 60813 |
| C1 | 7226 | 833 | 0 | 8059 |
| C2 | 2761 | 253 | 0 | 3014 |
| native | 0 | 929 | 0 | 929 |

## Edits per operation

| Operation | Edits |
| --- | ---: |
| M (missing) | 29174 |
| R (replacement) | 77713 |
| U (unnecessary) | 13629 |

## Top 15 error categories (all corpora)

| Category | Edits | Share |
| --- | ---: | ---: |
| PUNCT | 17520 | 14.5% |
| OTHER | 16091 | 13.4% |
| DET | 13679 | 11.4% |
| PREP | 12879 | 10.7% |
| VERB | 7828 | 6.5% |
| SPELL | 7821 | 6.5% |
| VERB:TENSE | 7518 | 6.2% |
| NOUN | 5511 | 4.6% |
| ORTH | 4937 | 4.1% |
| NOUN:NUM | 4589 | 3.8% |
| VERB:FORM | 4375 | 3.6% |
| PRON | 3705 | 3.1% |
| VERB:SVA | 2366 | 2.0% |
| MORPH | 2335 | 1.9% |
| ADV | 2099 | 1.7% |

## Error category by CEFR level (all corpora)

| Category | A1 | A2 | B1 | B2 | C1 | C2 | native | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PUNCT | 1327 | 3406 | 2862 | 7053 | 1715 | 803 | 354 | 17520 |
| OTHER | 1447 | 3238 | 2107 | 8056 | 873 | 271 | 99 | 16091 |
| DET | 1039 | 2393 | 1976 | 6925 | 968 | 340 | 38 | 13679 |
| PREP | 789 | 1993 | 1755 | 6989 | 925 | 363 | 65 | 12879 |
| VERB | 554 | 1336 | 976 | 4310 | 486 | 150 | 16 | 7828 |
| SPELL | 448 | 893 | 612 | 5400 | 276 | 74 | 118 | 7821 |
| VERB:TENSE | 682 | 1421 | 1019 | 3774 | 444 | 163 | 15 | 7518 |
| NOUN | 347 | 905 | 789 | 2893 | 395 | 150 | 32 | 5511 |
| ORTH | 816 | 1246 | 638 | 1887 | 197 | 91 | 62 | 4937 |
| NOUN:NUM | 380 | 923 | 767 | 2120 | 292 | 91 | 16 | 4589 |
| VERB:FORM | 346 | 832 | 639 | 2187 | 288 | 73 | 10 | 4375 |
| PRON | 247 | 666 | 446 | 2078 | 196 | 65 | 7 | 3705 |
| VERB:SVA | 252 | 528 | 391 | 993 | 135 | 64 | 3 | 2366 |
| MORPH | 154 | 401 | 336 | 1192 | 178 | 49 | 25 | 2335 |
| ADV | 135 | 334 | 272 | 1181 | 127 | 41 | 9 | 2099 |
| WO | 104 | 301 | 286 | 1172 | 156 | 73 | 4 | 2096 |
| ADJ | 121 | 296 | 261 | 908 | 143 | 57 | 11 | 1797 |
| PART | 69 | 186 | 132 | 228 | 101 | 29 | 3 | 748 |
| CONJ | 51 | 126 | 85 | 401 | 37 | 13 | 6 | 719 |
| NOUN:POSS | 56 | 95 | 101 | 325 | 83 | 33 | 23 | 716 |
| CONTR | 38 | 71 | 52 | 196 | 19 | 8 | 6 | 390 |
| NOUN:INFL | 18 | 24 | 18 | 272 | 8 | 5 | 4 | 349 |
| ADJ:FORM | 30 | 51 | 44 | 170 | 14 | 6 | 0 | 315 |
| VERB:INFL | 8 | 10 | 4 | 103 | 3 | 2 | 3 | 133 |

## Error category by CEFR level (W&I+LOCNESS dev, the gold-set pool)

| Category | A1 | A2 | B1 | B2 | C1 | C2 | native | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PUNCT | 155 | 306 | 228 | 219 | 156 | 60 | 354 | 1478 |
| OTHER | 152 | 285 | 198 | 122 | 106 | 18 | 99 | 980 |
| DET | 117 | 200 | 151 | 145 | 109 | 36 | 38 | 796 |
| PREP | 82 | 182 | 159 | 113 | 113 | 26 | 65 | 740 |
| VERB:TENSE | 74 | 165 | 115 | 61 | 34 | 9 | 15 | 473 |
| VERB | 54 | 109 | 92 | 64 | 48 | 19 | 16 | 402 |
| SPELL | 34 | 114 | 64 | 22 | 28 | 7 | 118 | 387 |
| ORTH | 65 | 116 | 63 | 24 | 12 | 10 | 62 | 352 |
| NOUN | 29 | 90 | 71 | 41 | 47 | 18 | 32 | 328 |
| NOUN:NUM | 41 | 88 | 61 | 21 | 16 | 8 | 16 | 251 |
| VERB:FORM | 31 | 68 | 61 | 31 | 29 | 6 | 10 | 236 |
| PRON | 26 | 59 | 30 | 34 | 17 | 5 | 7 | 178 |
| MORPH | 9 | 36 | 34 | 29 | 20 | 5 | 25 | 158 |
| VERB:SVA | 19 | 42 | 39 | 20 | 19 | 6 | 3 | 148 |
| ADV | 7 | 37 | 20 | 22 | 17 | 3 | 9 | 115 |
| ADJ | 6 | 28 | 28 | 19 | 18 | 3 | 11 | 113 |
| WO | 14 | 17 | 24 | 15 | 17 | 4 | 4 | 95 |
| NOUN:POSS | 8 | 4 | 10 | 7 | 10 | 4 | 23 | 66 |
| PART | 13 | 22 | 9 | 2 | 9 | 2 | 3 | 60 |
| CONJ | 6 | 14 | 8 | 5 | 3 | 2 | 6 | 44 |
| CONTR | 6 | 7 | 9 | 1 | 1 | 0 | 6 | 30 |
| ADJ:FORM | 3 | 5 | 3 | 3 | 2 | 0 | 0 | 16 |
| NOUN:INFL | 0 | 1 | 1 | 0 | 2 | 2 | 4 | 10 |
| VERB:INFL | 1 | 1 | 0 | 0 | 0 | 0 | 3 | 5 |
