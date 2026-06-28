# Final Report

Files:

- `final_report.tex` - LaTeX source.
- `final_report.pdf` - compiled report.
- `figures/` - copied figures used by the report.

To rebuild:

```powershell
cd report
pdflatex -interaction=nonstopmode -halt-on-error final_report.tex
pdflatex -interaction=nonstopmode -halt-on-error final_report.tex
```
