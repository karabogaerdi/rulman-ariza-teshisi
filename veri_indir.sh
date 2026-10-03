#!/usr/bin/env bash
# CWRU rulman verisinin sadece bu çalışmada kullanılan kısmını indirir (~380 MB).
# Kaynak: Case Western Reserve University Bearing Data Center'ın GitHub aynası.
set -e
mkdir -p data && cd data
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/s-whynot/CWRU-dataset
cd CWRU-dataset
git sparse-checkout set --no-cone \
  '/48k_Drive_End_Bearing_Fault_Data/B/' '/48k_Drive_End_Bearing_Fault_Data/IR/' \
  '/48k_Drive_End_Bearing_Fault_Data/OR/007/@6/' '/48k_Drive_End_Bearing_Fault_Data/OR/014/' \
  '/48k_Drive_End_Bearing_Fault_Data/OR/021/@6/' '/Normal/'
git checkout main
echo "Tamam: $(find . -name '*.mat' | wc -l) dosya indirildi (beklenen: 40)"
