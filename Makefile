.PHONY: help book clean serve

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  install     to install the necessary dependencies for jupyter-book to build"
	@echo "  book        to convert the content/ folder into book format in _book/"
	@echo "  clean       to clean out site build files"
	@echo "  runall      to run all notebooks in-place, capturing outputs with the notebook"
	@echo "  sync_md     to sync content in ipynb files from md files"
	@echo "  index       create an index file from _config/intro.md"

sync_md:
	./scripts/sync_md.sh

index:
	jb page _book/intro.md && echo "<meta http-equiv=\"Refresh\" content=\"0; url=intro.html\" />" > _book/_build/html/index.html

book:
	./scripts/make_book.sh

runall:
	jupyter-book run ./content

clean:
	rm -r _book/*

push: 
	ghp-import -n -p -f _book/_build/html

