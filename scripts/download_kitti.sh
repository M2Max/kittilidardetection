#!/usr/bin/env bash
# Scarica i dati KITTI necessari alla pipeline LiDAR-only.
# Solo velodyne + calib + label_2 + devkit. Nessuna immagine: per la valutazione
# quantitativa bastano le coordinate dei bbox 2D dentro label_2.
#
# Uso:  bash scripts/download_kitti.sh
# Ri-eseguibile: usa la resume di curl, salta cio' che e' gia' completo.

set -euo pipefail

BASE="https://s3.eu-central-1.amazonaws.com/avg-kitti"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DL="$ROOT/data/_archives"
DEST="$ROOT/data/kitti"

mkdir -p "$DL" "$DEST"

FILES=(
  "data_object_velodyne.zip"   # 28.75 GB - i point cloud .bin
  "data_object_calib.zip"      # 27 MB   - P2, R0_rect, Tr_velo
  "data_object_label_2.zip"    # 5.6 MB  - ground truth
  "devkit_object.zip"          # 64 KB   - evaluate_object.cpp
)

for f in "${FILES[@]}"; do
  dest="$DL/$f"
  expected=$(curl -sIL "$BASE/$f" | awk 'tolower($1)=="content-length:"{n=$2} END{print n+0}')

  if [[ -f "$dest" && $(stat -f%z "$dest") -eq $expected ]]; then
    echo "[skip] $f  (gia' completo, $(numfmt --to=iec $expected))"
  else
    echo "[dl]   $f  ->  $(numfmt --to=iec $expected)"
    curl -L -C - --retry 5 --retry-delay 5 -o "$dest" "$BASE/$f"
  fi
done

echo "[unzip] estrazione in $DEST"
for f in "${FILES[@]}"; do
  echo "  - $f"
  unzip -n -q "$DL/$f" -d "$DEST"
done

echo
echo "=== verifica ==="
n_velo=$(find "$DEST/training/velodyne" -name '*.bin' 2>/dev/null | wc -l | tr -d ' ')
n_lab=$(find "$DEST/training/label_2"   -name '*.txt' 2>/dev/null | wc -l | tr -d ' ')
n_cal=$(find "$DEST/training/calib"    -name '*.txt' 2>/dev/null | wc -l | tr -d ' ')
echo "velodyne: $n_velo   (attesi 7481)"
echo "label_2 : $n_lab   (attesi 7481)"
echo "calib   : $n_cal   (attesi 7481)"
[[ "$n_velo" -eq 7481 ]] || echo "ATTENZIONE: velodyne incompleti"