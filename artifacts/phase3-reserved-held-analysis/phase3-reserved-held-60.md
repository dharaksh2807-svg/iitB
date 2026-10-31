# phase3-reserved-held-60: held_query

|Configuration|n|All-facts %|Canonical EM %|KV MiB|Prefill ms|Query ms|TTFT ms|Wall ms|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|entity_balanced/B260/H128|60|76.7|35.0|7.1094|821.8|1546.4|2383.8|3213.3|
|entity_reserved/B260/H128|60|80.0|43.3|7.1094|840.4|1541.4|2397.4|3180.1|
|full|60|83.3|36.7|42.5327|799.5|1749.4|2552.5|3440.6|
|ogc/B260/C64|60|35.0|8.3|7.1094|847.9|1550.9|2412.4|3192.1|

Query/selection/decode durations overlap component totals; do not add selection to wall time.
