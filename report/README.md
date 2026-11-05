# Report

The short paper on the project, in the ACL format.

| File | Contents |
| --- | --- |
| `main.tex` | The paper |
| `custom.bib` | References |
| `acl.sty`, `acl_natbib.bst` | Official ACL style files, unchanged |
| `figures/` | Figures; `figures/grammar_graph.png` is drawn by `scripts/draw_graph.py` |

The style files come from
[acl-org/acl-style-files](https://github.com/acl-org/acl-style-files) at commit
`d5adc823ff0f80f98c80405ca0ab66c68e684409` (29 June 2026). The ACL asks authors not to
modify them; to update them, copy the new versions from that repository.

## Build

With a TeX distribution such as TeX Live:

```bash
cd report
latexmk -pdf main.tex
```

Or upload the folder to Overleaf and compile `main.tex` with pdfLaTeX.

`main.tex` uses the `preprint` option: not anonymous, with page numbers. Use `review`
for an anonymous submission and `final` for a camera-ready version.
