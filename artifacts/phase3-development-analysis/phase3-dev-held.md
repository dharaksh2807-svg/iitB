# phase3-dev-held: held_query

|Configuration|n|All-facts %|Canonical EM %|KV MiB|Prefill ms|Query ms|TTFT ms|Wall ms|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|entity_balanced/B260/H128|4|100.0|100.0|7.1094|364.7|1558.6|1933.5|2228.0|
|entity_chunks/B260|4|0.0|0.0|7.1094|371.7|1528.7|1917.4|2623.1|
|full|4|100.0|50.0|28.5332|361.4|1674.8|2038.9|2865.8|
|ogc/B260/C64|4|0.0|0.0|7.1094|372.8|1563.0|1944.7|2416.2|
|ogc_c32/B260/C32|4|0.0|0.0|7.1094|367.0|1530.8|1907.0|2613.5|
|ogc_corrected/B260/C64|4|0.0|0.0|7.1094|377.8|1544.9|1931.8|2401.2|
|ogc_history/B260/C64/H128|4|25.0|25.0|7.1094|364.7|1548.7|1922.3|2191.5|

Query/selection/decode durations overlap component totals; do not add selection to wall time.
