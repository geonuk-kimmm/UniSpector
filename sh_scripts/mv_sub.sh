# -print0 and xargs -0 safely handle file names containing spaces.
find . -maxdepth 1 -name "*.tar" -print0 | xargs -0 -n 1 -P "${NUM_PROCS}" bash -c 'extract_and_remove "$@"' _

echo "All archives have been extracted. Data is in '/aidata01/geonuk/extracted_data'."

cd /aidata01/geonuk/extracted_data/

for file in sa_*.jpg sa_*.json; do
    # Extract the first 9 characters from filename (e.g., sa_00000001 -> sa_000000)
    prefix=${file:0:9}
    mkdir -p "$prefix"
    mv "$file" "$prefix/"
done 