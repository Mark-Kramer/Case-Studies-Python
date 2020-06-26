VPATH = binder
STARTUP = 00-CSNDA-imports.py 01-CSNDA-styling.py
CUSTOM = custom.js custom.css

CUSTOMDIR = ~/.jupyter/custom/
STARTUPDIR = ~/.ipython/profile_default/startup/
LABDIR = $$CONDA_PREFIX/share/jupyter/lab/themes/\@jupyterlab/theme-light-extension/

PKG = Case Studies in Neural Data Analysis

# ------------------------------------------------------------
.PHONY: help env active config clean cleanenv

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  env         Creates the environment <case-studies>"
	@echo "  active      Activates the environment <case-studies>"
	@echo "  config      Mimics the notebook configuration used in Case Studies in Neuroscience"
	@echo "  clean       Removes config content from Jupyter path"
	@echo "  cleanenv    Removes the environment <case-studies>"
# ------------------------------------------------------------


# ------------------------------------------------------------
# Move the CUSTOM and STARTUP files to their destination directories
# so that they are recognized by Jupyter
 
custom = $(addprefix $(CUSTOMDIR), $(CUSTOM))
lab = $(addprefix $(LABDIR), index.css)
startup = $(addprefix $(STARTUPDIR), $(STARTUP))

# Set up comments (css: /* ... */; js: // ... //)
%.css: C = *
%.js: C = /

# Remove and replace custom content
%: F = $(VPATH)/$(@F)
$(lab): F = $(VPATH)/custom.css
$(custom) $(lab): $(CUSTOM) | $(CUSTOMDIR) $(LABDIR)  
	@touch $@ && sed -i.old '/START: $(PKG)/,/END: $(PKG)/d' $@
	@echo "/$C ------- START: $(PKG) ----- $C/" >> $@
	cat $(F) >> $@ 
	@echo "/$C ----- END: $(PKG) ----- $C/" >> $@ 

$(startup): $(STARTUP) | $(STARTUPDIR)
	cp $(F) $@  # copy the local file to the destination
	
$(STARTUPDIR) $(CUSTOMDIR) $(LABDIR):
	@mkdir -p $@  # make the target directory
# ------------------------------------------------------------


# ------------------------------------------------------------
# Create conda environment
env: environment.yml
	conda env create --file $< 

# Activate conda environment
active:
	conda activate case-studies

# Make CUSTOM and STARTUP files visible to Jupyter
config: $(custom) $(startup) $(lab)

# remove STARTUP files and delete CUSTOM content from Jupyter path
clean:  
	rm -f $(startup)
	-sed -i.old '/START: $(PKG)/,/END: $(PKG)/d' $(custom) $(lab)
	@touch $(CUSTOM)

# Delete the conda environment case-studies
cleanenv:
	conda deactivate
	conda remove --name case-studies --all


