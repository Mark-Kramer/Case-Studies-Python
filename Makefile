.PHONY: help book clean binder runall install site book sync_md all

help:
	@echo ""
	@echo "=== USAGE ==="
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  install     Instructions for creating the 'csn' environment"
	@echo "  book        to convert the content/ folder into book format in _book/"
	@echo "  clean       to clean out site build files"
	@echo "  runall      to run all notebooks in-place, capturing outputs with the notebook"
	@echo "  sync_md     to sync content in ipynb files from md files"
	@echo "  index       create an index file from _config/intro.md"
	@echo "  site        push site in _build/html to gh-pages branch"
	@echo "  binder      update the student branch with all files in _toc.yml, matfiles, imgs, *.py, README.md, and anything in a folder named student"
	@echo ""
	@echo "*** After updating a chapter, you will want to run 'make book site binder' or 'make all' within the 'csn' environment (identical to the 'case-studies' environment, but contains additional packages required for book upkeep). This will update the html files using the jupyter-book package and the push all updates to the appropriate repo branches on github. The content hosted through Binder is pushed to the 'binder' branch using 'make binder', while the content hosted as a GitHub site is pushed to the 'gh-pages' branch using 'make site'. The purpose of the 'binder' branch is to isolate the book content from the book upkeep so that students using Binder can focus on the content."
	@echo "============="
	@echo ""

all: clean book site binder

binder:
	./_config/scripts/make_binder.sh

%.ipynb: %.md
	./_config/scripts/sync_md.sh $?

sync_md: *.ipynb

book:
	./_config/scripts/make_book.sh

runall:
	jupyter nbconvert --to notebook --execute --inplace -y --ExecutePreprocessor.timeout=-1 *.ipynb	

install:
	@echo "Run 'conda env create --file _config/csn.yml' to create the 'csn' environment which has all packages for running book content as well as jupyter-book for building the book html."

clean:
	rm -rf _build

site: 
	ghp-import -n -p -f _build/html

