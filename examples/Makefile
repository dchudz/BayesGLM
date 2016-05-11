all: output/binomial.ipynb

output/binomial.ipynb: source/binomial.ipynb
	jupyter nbconvert --ExecutePreprocessor.enabled=True c.ExecutePreprocessor.timeout=600 $^  --to notebook --output=$@
