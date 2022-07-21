for filename in *.jpg; do
filename_base=$(echo "$filename" | cut -f 1 -d '.')
echo ________ $filename $filename_base ___________________
  convert "$filename" -resize 300x423  "scaled/${filename_base}_scaled.jpg"
done

