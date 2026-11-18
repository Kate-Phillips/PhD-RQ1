Smoke test: python experiments.py --pilot --seeds 1 --epochs 5 <br />
Confirms the pipeline runs end-to-end on 2 datasets, 1 seed, & 5 epochs 

Shuffle pilot: python experiments.py --pilot --seeds 3 --epochs 50 <br />
               python analyze_results.py --results results.csv <br />
Looks specifically at the P3 shuffle output. Falsification test.

Full batch: python experiments.py --out results_full.csv --seeds 5 --epochs 100 <br />
            python analyze_results.py --results results_full.csv <br />
