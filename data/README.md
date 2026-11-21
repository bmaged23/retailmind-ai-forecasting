# Data acquisition

Online Retail II is provided by the UCI Machine Learning Repository: Chen, D. (2012), *Online Retail II*, [DOI 10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The workbook contains transaction lines for a UK retailer in 2009–2011, not inventory levels or observed lost demand.

The original UCI archive was downloaded from https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip. Extract `online_retail_II.xlsx` into `data/raw/`. The ZIP integrity test succeeded. The workbook SHA-256 is `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`, matching an [independent published snapshot](https://github.com/DiogoRibeiro7/data/blob/main/datasets/online-retail-ii/metadata.yaml). It has two sheets: Year 2009-2010 (525,461 data rows) and Year 2010-2011 (541,910 data rows), totaling 1,067,371 rows. Counts are workbook sheet dimensions, excluding the headers.

Raw data and generated data are ignored by Git. Run `uv run retailmind prepare data/raw/online_retail_II.xlsx` to create the daily panel and cleaning audit. See the root README for limitations of the current cleaning policy.

## Visual reports

Open the [data exploration dashboard](../reports/data-exploration.html), [feature analysis](../reports/feature-eda.html), and [product-code audit](../reports/code-family-audit.html) in a browser. These are retained in the repository; raw and processed datasets remain excluded from Git. See the [report index](../reports/README.md) for the full set.
