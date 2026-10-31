# phase3-dev-continuation: continuation

|Configuration|n|All-facts %|Canonical EM %|KV MiB|Prefill ms|Query ms|TTFT ms|Wall ms|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|entity_balanced/B260/H128|4|100.0|50.0|7.1094|382.5|0.0|392.8|1127.1|
|entity_chunks/B260|4|75.0|25.0|7.1094|385.2|0.0|398.1|1226.7|
|full|4|100.0|50.0|28.5332|374.8|0.0|377.2|1157.5|
|ogc/B260/C64|4|0.0|0.0|7.1094|378.5|0.0|386.8|1070.7|
|ogc_c32/B260/C32|4|0.0|0.0|7.1094|381.0|0.0|389.5|1141.5|
|ogc_corrected/B260/C64|4|0.0|0.0|7.1094|433.3|0.0|444.0|1134.1|
|ogc_history/B260/C64/H128|4|0.0|0.0|7.1094|379.2|0.0|387.9|1121.0|

Query/selection/decode durations overlap component totals; do not add selection to wall time.
