set terminal pngcairo size 1460,650 enhanced font 'Verdana,16'
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set yrange [0:*]
set ylabel "Average Power Consumption (W)"
set xlabel "Model"
set xtics rotate by -30
set key outside right #above #fixed top horizontal Right noreverse noenhanced autotitle nobox
set title "Jetson-Thor vs IC2 vs H200"
set label "llama = meta-llama/Meta-Llama-3.1-8B-Instruct" at screen 0.745, screen 0.420 left font 'Verdana,8' front
set label "gemma = google/gemma-4-E4B-it" at screen 0.745, screen 0.370 left font 'Verdana,8' front
set label "gpt = openai/gpt-oss-20b" at screen 0.745, screen 0.320 left font 'Verdana,8' front
set label "qwen = Qwen/Qwen2.5-7B-Instruct" at screen 0.745, screen 0.270 left font 'Verdana,8' front
set output 'power_combined.png'
plot 'combined_transformers_vs_vllm.dat' using 14:xtic(1) title 'JT vLLM', \
     '' using 15:xtic(1) title 'JT Transformers', \
     '' using 16:xtic(1) title 'IC2 vLLM', \
     '' using 17:xtic(1) title 'IC2 Transformers', \
     '' using 18:xtic(1) title 'H200 vLLM', \
     '' using 19:xtic(1) title 'H200 Transformers'
