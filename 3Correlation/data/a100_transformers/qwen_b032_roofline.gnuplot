set terminal pngcairo size 1500,1100 enhanced font 'Sans,11'
set output 'sweep_a100_tf/qwen_b032_roofline.png'
set datafile separator "\t"
set datafile missing 'NaN'
set multiplot layout 2,2
unset label
unset object
unset arrow
set title "Qwen2.5-7B-Instruct (batch 32) (empirical ceilings)" font ',13'
set xlabel "Arithmetic Intensity (FLOP/byte)"
set ylabel "Performance (TFLOP/s)"
set logscale xy
set xrange [1:3408.21]
set yrange [5.43285:456.547]
set grid xtics ytics
set key top left
set object 1 rectangle from graph 0, graph 0 to first 243.865, graph 1 \
    fillcolor rgb '#dce9f7' fillstyle solid 0.45 noborder behind
set object 2 rectangle from first 243.865, graph 0 to graph 1, graph 1 \
    fillcolor rgb '#fbdcdc' fillstyle solid 0.45 noborder behind
set label 1 "Memory Bound" at graph 0.04, graph 0.55 left \
    textcolor rgb '#2563eb' font ',11'
set label 2 "Compute Bound" at graph 0.96, graph 0.55 right \
    textcolor rgb '#dc2626' font ',11'
set label 3 "Ridge(243.9, 182.6)" \
    at first 243.865, first 182.619 right \
    point pointtype 5 pointsize 1.2 \
    offset character -1.5, character 1 font ',10'
set label 4 "decode 1373.0 tok/s, 621 GB/s (phase totals)\npower 269.1 W avg / 463.3 W peak\nutil 96% avg / 97% peak\n0.15 J/tok dyn (0.20 raw)" at graph 0.96, graph 0.16 right font ',10'
peak_t = 182.619
peak_b = 748.851
roof(x) = (x * peak_b / 1000.0 < peak_t) ? x * peak_b / 1000.0 : peak_t
plot roof(x) with lines linewidth 3 linecolor rgb '#111111' title 'Roof', \
     'sweep_a100_tf/qwen_b032_roofline.dat' using 3:(strcol(2) eq 'prefill' ? $4 : 1/0) with points \
         pointtype 5 pointsize 1.6 linecolor rgb '#1f77b4' title 'Prefill', \
     'sweep_a100_tf/qwen_b032_roofline.dat' using 3:(strcol(2) eq 'decode' ? $4 : 1/0) with points \
         pointtype 7 pointsize 1.0 linecolor rgb '#ff7f0e' title 'Decode'

unset label
unset object
unset arrow
unset logscale
set xrange [*:*]
set yrange [*:*]
set title "Qwen2.5-7B-Instruct (batch 32): decode throughput" font ',12'
set xlabel "step"
set ylabel "tokens/s"
set grid xtics ytics
set key top right font ',9'
plot 'sweep_a100_tf/qwen_b032_roofline.dat' using 1:(strcol(2) eq 'decode' ? $5 : 1/0) with points pointtype 7 pointsize 0.4 linecolor rgb '#ff7f0e' title 'tok/s'

unset label
unset object
unset arrow
unset logscale
set xrange [*:*]
set yrange [*:*]
set title "Qwen2.5-7B-Instruct (batch 32): GPU power" font ',12'
set xlabel "step"
set ylabel "watts"
set grid xtics ytics
set key top right font ',9'
plot 'sweep_a100_tf/qwen_b032_roofline.dat' using 1:(strcol(2) eq 'decode' ? $6 : 1/0) with points pointtype 7 pointsize 0.4 linecolor rgb '#d62728' title 'avg', \
     'sweep_a100_tf/qwen_b032_roofline.dat' using 1:(strcol(2) eq 'decode' ? $8 : 1/0) with points pointtype 7 pointsize 0.4 linecolor rgb '#f4a6a6' title 'peak'

unset label
unset object
unset arrow
unset logscale
set xrange [*:*]
set yrange [*:*]
set title "Qwen2.5-7B-Instruct (batch 32): GPU utilization" font ',12'
set xlabel "step"
set ylabel "%"
set grid xtics ytics
set key top right font ',9'
plot 'sweep_a100_tf/qwen_b032_roofline.dat' using 1:(strcol(2) eq 'decode' ? $7 : 1/0) with points pointtype 7 pointsize 0.4 linecolor rgb '#2ca02c' title 'avg', \
     'sweep_a100_tf/qwen_b032_roofline.dat' using 1:(strcol(2) eq 'decode' ? $9 : 1/0) with points pointtype 7 pointsize 0.4 linecolor rgb '#a9d7a9' title 'peak'

unset multiplot
