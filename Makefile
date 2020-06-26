.PHONY: help book clean serve binder

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  install     to install the necessary dependencies for jupyter-book to build"
	@echo "  book        to convert the content/ folder into book format in _book/"
	@echo "  clean       to clean out site build files"
	@echo "  runall      to run all notebooks in-place, capturing outputs with the notebook"
	@echo "  sync_md     to sync content in ipynb files from md files"
	@echo "  index       create an index file from _config/intro.md"
	@echo "  site        push site in _build/html to gh-pages branch"
	@echo "  binder      update the student branch with all files in _toc.yml, matfiles, imgs, *.py, README.md, and anything in a folder named student"

all: clean book site binder

binder:
	./_config/scripts/make_binder.sh

sync_md:
	./_config/scripts/sync_md.sh	

book:
	./_config/scripts/make_book.sh

runall:
	jupyter nbconvert --to notebook --execute --inplace -y --ExecutePreprocessor.timeout=-1 *.ipynb	

clean:
	rm -rf _build

site: 
	ghp-import -n -p -f _build/html

