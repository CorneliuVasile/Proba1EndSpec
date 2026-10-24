# Supplier category mapping

Map every original supplier category name to exactly one of the retailer's 150 destination categories. The final submission is `output/mapping.csv` with exactly `supplier_category,our_category_id` and one line per distinct original spelling.

## Run (Python 3.10+)

```bash
python -m pip install -r requirements.txt
python main.py
python -m unittest discover -s tests -v
python evaluate.py
python review.py
```

Place the four original inputs under `data/`. All commands run from the project root. `main.py` writes the submission; `evaluate.py` and `review.py` produce diagnostic CSVs under `output/` without changing the submission.

## Method and verification (short write-up)

The supplier's category names are normalized by removing whitespace and applying case-folding; the original spellings are preserved for export. Product SKUs connect the product table to the hand labels. For categories with labels, the destination is the most common hand-labelled destination, with a deterministic tie-break. For categories without labels, text from category names and up to 35 distinct product-name/brand examples becomes a TF-IDF representation (unigrams/bigrams). Cosine similarity assigns the destination of the nearest labelled supplier category. Three explicitly documented, manually reviewed overrides are applied at the **normalized category** level. The full pipeline is deterministic given fixed input order and library behavior. It requires no remote API or proprietary service.

We verified every original spelling occurs exactly once, all category IDs exist, and names equivalent under case/whitespace normalize to the same destination. A category-level 20% holdout (seed 42) compared a most-common-destination baseline, nearest-category TF-IDF, and a supervised TF-IDF RidgeClassifier. In this one split they agreed with held-out majority labels on 1/326 (0.31%), 276/326 (84.66%), and 284/326 (87.12%) respectively. Ridge did better in this test, but was **not automatically substituted** in the output because it also introduced some errors and the holdout is not independently verified ground truth. The production mapping reproduces the reviewed nearest-neighbour solution, rather than claiming the Ridge evaluation score as its production accuracy. Manual overrides were motivated by keyword/turnover inspection, not product-by-product adjudication.

We deliberately did not invent a per-category financial loss model, equate sales revenue with lost revenue, or assume similarity scores are calibrated confidence. We did not force all 106 nearest/Ridge disagreements to use Ridge. Weak spots include 1,051 normalized supplier categories with no hand labels, 299 categories with conflicting hand labels, mixed-content supplier categories, and possible bad hand labels. The review queue prioritizes high-revenue uncertain cases; **associated turnover is exposure, not loss**.

**AI use:** AI assisted in developing and reviewing Python scripts, analyses, and documentation; we checked the reported statistics, sample category contents, and output invariants by executing local code and spot-checking examples. This is not a claim of complete manual validation.

**Outside scope / not fixed:** inconsistent supplier category content; contradictory manual labels; possibly incorrect retail labels; lack of a product-level rerouting option under the one-destination-per-supplier-category rule.

## Files

- `main.py`: creates `output/mapping.csv` from source CSVs with three reviewed overrides.
- `evaluate.py`: category-level holdout and method comparison, creating `output/evaluation.csv`.
- `review.py`: revenue-based flagged-category report, creating `output/review_queue.csv`.
- `tests/test_mapping.py`: export and consistency checks.
- `data/*.csv`: the four source inputs.
- `output/mapping.csv`: submission candidate (not independently human verified in full).

Do not claim that the holdout accuracy guarantees accuracy on truly unseen supplier categories. The evaluation score and the output mapping use different fitting procedures: the export vectorizer fits vocabulary to all unlabelled category text, without their destination labels; the evaluation vectorizer learns vocabulary on train categories only.
