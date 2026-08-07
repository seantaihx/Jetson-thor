set terminal pngcairo size 1460,650 enhanced font 'Verdana,16'
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set yrange [0:*]
set ylabel "Throughput (tokens/s)"
set xlabel "Model"
set xtics rotate by -30
set key outside right noenhanced
set title "A100-SXM4-80GB - vLLM CUDA graphs vs enforce eager vs Transformers"
set label "llama = /hpcgpfs01/scratch/stai/models/Llama-3.1-8B-Instruct" at screen 0.725, screen 0.420 left font 'Verdana,9' front noenhanced
set label "gemma = /hpcgpfs01/scratch/stai/models/gemma-4-E4B-it" at screen 0.725, screen 0.370 left font 'Verdana,9' front noenhanced
set label "gpt = /hpcgpfs01/scratch/stai/models/gpt-oss-20b" at screen 0.725, screen 0.320 left font 'Verdana,9' front noenhanced
set label "qwen = /hpcgpfs01/scratch/stai/models/Qwen2.5-7B-Instruct" at screen 0.725, screen 0.270 left font 'Verdana,9' front noenhanced
set output 'throughput_eager3.png'
plot \
     'combined_eager3.dat' using 2:xtic(1) title 'vLLM (CUDA graphs)', \
     '' using 3:xtic(1) title 'vLLM (enforce_eager)', \
     '' using 4:xtic(1) title 'Transformers'
