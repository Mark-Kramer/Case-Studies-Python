.PHONY: env config help

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  env         Creates the environment <case-studies>"
	@echo "  config      Mimics the notebook configuration used in Case Studies in Neuroscience"

env:
	conda create env --file environment.yml
	
config:
	mkdir -p ~/.jupyter/custom
	cp -i _config/_static/custom.* ~/.jupyter/custom/
	mkdir -p ~/.ipython/profile_default/
	cp -ir _config/startup ~/.ipython/profile_default/

