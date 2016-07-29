pdf:
	pandoc gans.md -s \
		--filter pandoc-crossref \
		--filter pandoc-citeproc \
		-o gans_p.pdf