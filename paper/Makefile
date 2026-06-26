LATEXMK ?= latexmk

all: main.pdf main_zh.pdf

main.pdf: main.tex references.bib sections/*.tex tables/*.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

main_zh.pdf: main_zh.tex references.bib sections_zh/*.tex tables/*.tex
	$(LATEXMK) -xelatex -interaction=nonstopmode -halt-on-error main_zh.tex

en: main.pdf

zh: main_zh.pdf

clean:
	$(LATEXMK) -C
	$(LATEXMK) -C main_zh.tex

.PHONY: all en zh clean
