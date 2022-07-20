for filename in *.pdf; do
filename_base=$(echo "$filename" | cut -f 1 -d '.')
echo ________ $filename $filename_base ___________________
  convert -density 500 -colorspace rgb "$filename" -scale 100% "img/${filename_base}_image.jpg"
done
