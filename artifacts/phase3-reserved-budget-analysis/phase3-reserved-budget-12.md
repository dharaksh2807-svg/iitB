# phase3-reserved-budget-12: held_query

|Configuration|n|All-facts %|Canonical EM %|KV MiB|Prefill ms|Query ms|TTFT ms|Wall ms|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|entity_balanced/B1028/H512|12|75.0|33.3|28.1094|3267.7|1733.3|5020.8|6028.4|
|entity_balanced/B260/H128|12|83.3|41.7|7.1094|3308.6|1560.2|4891.0|5720.9|
|entity_balanced/B516/H256|12|75.0|33.3|14.1094|3237.6|1636.7|4899.3|5846.0|
|entity_reserved/B1028/H512|12|75.0|33.3|28.1094|3198.8|1702.8|4921.5|5923.2|
|entity_reserved/B260/H128|12|83.3|41.7|7.1094|3282.8|1581.6|4890.9|5729.4|
|entity_reserved/B516/H256|12|83.3|33.3|14.1094|3260.5|1618.8|4901.1|5861.9|
|full|12|83.3|33.3|105.6243|3225.8|1997.7|5229.1|6447.5|
|ogc/B1028/C64|12|33.3|8.3|28.1094|3262.1|1755.3|5032.0|6126.9|
|ogc/B260/C64|12|16.7|0.0|7.1094|3237.3|1550.1|4803.7|5706.6|
|ogc/B516/C64|12|16.7|0.0|14.1094|3248.9|1638.3|4904.5|5749.5|

Query/selection/decode durations overlap component totals; do not add selection to wall time.
