
# ===========================
# scripts/download_cihp.sh
# ===========================
#!/usr/bin/env bash
set -e
mkdir -p data/cihp && cd data/cihp
echo "Please download CIHP dataset manually (license). Place images under Images/{train,val} and masks under Annotations/{train,val}."
