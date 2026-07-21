.PHONY: probe test red green clean
# Build the frozen-probe upload zip.
probe:
	cd probes/reve_frozen_probe && ./make_zip.sh
# Run the TDD contract (RED before an NSG run, GREEN after fetching results).
test:
	pytest probes/tests -q
red: test
green: test
clean:
	rm -rf probes/**/results probes/**/*.zip templates/**/*.zip templates/**/vendor
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
