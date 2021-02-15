config=$1
target=${2:-kwdlc_pred}

for case in "ga" "wo" "ni" "ga2"
do
    make test CONFIG=$config TARGET=$target CASE=$case
done