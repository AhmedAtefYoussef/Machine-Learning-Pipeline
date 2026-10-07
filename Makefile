# Rush Hour pipeline. Phase N rebuilds automatically when anything upstream changes (chain integrity by construction).
# Output is deliberately compact: every check prints a one-line summary (token budget).
PYRUN = PYTHONPATH=. python3
TOOLS = python3 -I tools
ART = artifacts
HAVE = $(shell ls $(ART)/p?.json 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/.json//' | paste -sd, -)
SCRATCH = --scratch-glob 'nb/p1*.py' --scratch-glob 'nb/p2*.py' --scratch-glob 'src/gd*.py' --scratch-glob 'src/phases/p1.py' --scratch-glob 'src/phases/p2.py'

.PHONY: all verify-fast verify nb report fingerprint kit-check clean
all: $(ART)/p5.json

$(ART)/p1.json: config.yaml src/common.py src/gd.py src/phases/p1.py
	$(PYRUN) -m src.phases.p1
$(ART)/p2.json: $(ART)/p1.json src/poly.py src/phases/p2.py
	$(PYRUN) -m src.phases.p2
$(ART)/p3.json: $(ART)/p2.json src/validation.py src/phases/p3.py
	$(PYRUN) -m src.phases.p3
$(ART)/p4.json: $(ART)/p3.json src/regularization.py src/phases/p4.py
	$(PYRUN) -m src.phases.p4
$(ART)/p5.json: $(ART)/p4.json src/logistic.py src/phases/p5.py
	$(PYRUN) -m src.phases.p5

verify-fast:
	@-(ruff check src tools tests 2>&1 || true) | tail -n 15
	@$(PYRUN) -m pytest -q -x tests 2>&1 | tail -n 15
	@$(TOOLS)/leak_scan.py src nb $(SCRATCH) | tail -n 15
	@if [ -n "$(HAVE)" ]; then $(TOOLS)/chain_check.py --require $(HAVE) | tail -n 15; else echo "chain_check: no artifacts yet"; fi
	@$(TOOLS)/req_check.py | tail -n 10

verify: verify-fast
	@$(TOOLS)/build_nb.py --execute --out rush_hour.ipynb | tail -n 15
	@$(TOOLS)/submission_check.py | tail -n 10
	@$(TOOLS)/number_trace.py render report/report.template.md report/report.md | tail -n 5
	@$(TOOLS)/number_trace.py check report/report.md | tail -n 15
	@$(TOOLS)/req_check.py --strict | tail -n 15

nb:
	@$(TOOLS)/build_nb.py --execute --out rush_hour.ipynb | tail -n 15

report:
	@$(TOOLS)/number_trace.py render report/report.template.md report/report.md
	@pandoc report/report.md -o report/report.pdf 2>&1 | tail -n 5 || echo "pandoc missing: use the pdf skill"

fingerprint:
	@$(TOOLS)/fingerprint.py $(ART)

kit-check:
	@test -f CLAUDE.md && ls .claude/agents/*.md .claude/skills/*/SKILL.md | wc -l | xargs echo "kit files:" ; grep -L '^name:' .claude/agents/*.md .claude/skills/*/SKILL.md || echo "frontmatter ok"

clean:
	rm -rf $(ART)/p?.json rush_hour.ipynb report/report.md
