for N in "table_cifar" "table_cifar100" "table_lsuncat"
do
    python collate_markdown_table.py --input_json "OUT/$N.json" \
        --output_txt "OUT/$N.txt"
done

for N in "table_afhq" "table_brecahad" "table_metfaces"
do
    python collate_markdown_table.py --input_json "OUT/$N.json" \
        --output_txt "OUT/$N.txt" --use_kid
done