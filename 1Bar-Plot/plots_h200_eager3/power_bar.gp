set terminal pngcairo size 1460,650 enhanced font 'Verdana,16'
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set yrange [0:*]
set ylabel "Average Dynamic Power (W, measured - idle)"
set xlabel "Model"
set xtics rotate by -30
set key outside right noenhanced
set title "H200 NVL - vLLM CUDA graphs vs enforce eager vs Transformers"
set label "llama = meta-llama/Meta-Llama-3.1-8B-Instruct" at screen 0.725, screen 0.420 left font 'Verdana,9' front noenhanced
set label "gemma = google/gemma-4-E4B-it" at screen 0.725, screen 0.370 left font 'Verdana,9' front noenhanced
set label "gpt = openai/gpt-oss-20b" at screen 0.725, screen 0.320 left font 'Verdana,9' front noenhanced
set label "qwen = Qwen/Qwen2.5-7B-Instruct" at screen 0.725, screen 0.270 left font 'Verdana,9' front noenhanced
set output 'power_eager3.png'
plot \
     'combined_eager3.dat' using 8:xtic(1) title 'vLLM (CUDA graphs)', \
     '' using 9:xtic(1) title 'vLLM (enforce_eager)', \
     '' using 10:xtic(1) title 'Transformers'
