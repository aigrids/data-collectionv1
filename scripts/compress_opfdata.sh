#!/bin/bash
: '
Compress processed OPFData files for upload to Hugging Face.

Example run:
    $ ./compress_opfdata.sh regular pglib_opf_case14_ieee

or:
    $ ./compress_opfdata.sh n-1 pglib_opf_case14_ieee

List of cases:

    pglib_opf_case14_ieee
    pglib_opf_case30_ieee
    pglib_opf_case57_ieee
    pglib_opf_case118_ieee
    pglib_opf_case500_goc
    pglib_opf_case2000_goc
    pglib_opf_case4661_sdet
    pglib_opf_case6470_rte
    pglib_opf_case10000_goc
    pglib_opf_case13659_pegase

'
# base path to processed OPFData repository
PATH_ORIGIN=<set-path-to-source>
PATH_TARGET=<set-path-to-target>

echo "realease": $1
echo "grid": $2

# create path to release
if [ $1 == "regular" ]; then
    PATH_ORIGIN="${PATH_ORIGIN}/dataset_release_1"
elif [ $1 == "n-1" ]; then
    PATH_ORIGIN="${PATH_ORIGIN}/dataset_release_1_nminusone"
else
    echo "invalid first argument"
    exit 1
fi

# create path to dataset
PATH_ORIGIN="${PATH_ORIGIN}/$2"
PATH_TARGET="${PATH_TARGET}/$2"

# Create target directory if needed
mkdir -p "$PATH_TARGET"

for entry in $(ls "$PATH_ORIGIN"); do
    DIR_NAME="$entry"
    PATH_FILENAME="${PATH_TARGET}/${DIR_NAME}.tar.gz"
    tar -czvf "$PATH_FILENAME" -C "$PATH_ORIGIN" "$DIR_NAME" &
done 

# Wait until all parallel background processes complete
wait

echo "Compression script ran successfully!"