set terminal pngcairo size 1200,600 enhanced font 'Verdana,16'
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set yrange [0:*]
set ylabel "Average Power Consumption (W)"
set xlabel "Model"
set xtics rotate by -30
set key outside right #above #fixed top horizontal Right noreverse noenhanced autotitle nobox
set title "Nvidia Jetson-thor"
set label "llama = meta-llama/Meta-Llama-3.1-8B-Instruct" at screen 0.745, screen 0.420 left font 'Verdana,8' front
set label "gemma = google/gemma-4-E4B-it" at screen 0.745, screen 0.370 left font 'Verdana,8' front
set label "gpt = openai/gpt-oss-20b" at screen 0.745, screen 0.320 left font 'Verdana,8' front
set output 'power_transformers_vs_vllm.png'
plot 'transformers_vs_vllm.dat' using 6:xtic(1) title 'vLLM', \
     '' using 7:xtic(1) title 'Transformers'
