#!/bin/bash

# Cartella contenente i file .csv
csv_folder="PostProcessing"

# Cartella contenente i file .pth
pth_folder="Features6"

# Trova i nomi dei file .csv nella cartella csv_folder
csv_files=$(ls "${csv_folder}"/ResPrecisionRecall_*.csv 2>/dev/null)

# Trova e elimina i file .pth nella cartella pth_folder se il nome coincide con i file .csv
for csv_file in $csv_files; do
    # Estrai il nome del file .csv senza estensione e senza percorso
    csv_filename=$(basename -- "$csv_file" .csv)
    # Rimuovi il prefisso "ResPrecisionRecall_" dal nome del file .csv
    csv_filename=${csv_filename#ResPrecisionRecall_}
    # Estrai il nome del file .pth corrispondente
    pth_file="${pth_folder}/${csv_filename}.pth"
    # Verifica se esiste il file .pth e lo elimina
    if [ -e "$pth_file" ]; then
        rm -f "$pth_file"
        echo "Il file $pth_file è stato eliminato."
    else
        echo "Attenzione: Non esiste il file $pth_file corrispondente al file $csv_file."
    fi
done
