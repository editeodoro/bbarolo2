# Build the documentation: make html (or make pdf, make linkcheck, make clean)
SPHINXBUILD ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

html:
	$(SPHINXBUILD) -W --keep-going -n -b html $(SOURCEDIR) $(BUILDDIR)/html

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck

pdf:
	$(SPHINXBUILD) -M latexpdf $(SOURCEDIR) $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)

.PHONY: html linkcheck pdf clean
