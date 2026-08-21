# =============================================================================
#  Machines That Learn and Act — build targets
#
#    make html          render the HTML site into build/html
#    make pdf           render every part to build/pdf/part-NN.pdf
#    make pdf PART=07   render one part only
#    make book          merge the part PDFs into one volume
#    make code          extract every tagged code block into code/
#    make check         run all quality gates (structure, refs, code, citations)
#    make report        per-chapter status dashboard
#    make all           code + html + pdf + book
#    make serve         preview the site at http://localhost:8000
#    make clean         remove build output (keeps the render caches)
#    make distclean     remove build output and caches
# =============================================================================

PY      := .venv/bin/python
TOOLS   := tools
PART    ?=

.PHONY: all html pdf book code check report serve clean distclean deps

all: code html pdf book

html:
	@$(PY) $(TOOLS)/build.py html

pdf:
ifeq ($(strip $(PART)),)
	@$(PY) $(TOOLS)/build.py pdf
else
	@$(PY) $(TOOLS)/build.py pdf --part $(PART)
endif

book:
	@$(PY) $(TOOLS)/build.py merge

code:
	@$(PY) $(TOOLS)/build.py code

check:
	@$(PY) $(TOOLS)/check.py

report:
	@$(PY) $(TOOLS)/report.py

# Render, check, and build one part end to end — the per-part workflow.
part:
	@test -n "$(PART)" || (echo "usage: make part PART=07" && exit 1)
	@$(PY) $(TOOLS)/build.py code
	@$(PY) $(TOOLS)/check.py --part $(PART)
	@$(PY) $(TOOLS)/build.py html
	@$(PY) $(TOOLS)/build.py pdf --part $(PART)

serve: html
	@echo "http://localhost:8000/"
	@cd build/html && ../../$(PY) -m http.server 8000

deps:
	@$(PY) -m pip install -q -r requirements.txt
	@PUPPETEER_SKIP_DOWNLOAD=1 npm install --no-audit --no-fund

clean:
	@rm -rf build/html build/pdf
	@echo "removed build/html and build/pdf (caches kept)"

distclean: clean
	@rm -rf build/cache
	@echo "removed render caches"
