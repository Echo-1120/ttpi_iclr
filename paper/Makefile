LATEXMK ?= latexmk

all: main.pdf

main.pdf: main.tex references.bib sections/*.tex tables/*.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(LATEXMK) -C

.PHONY: all clean
