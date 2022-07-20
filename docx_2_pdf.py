from docx2pdf import convert
import os

def convert_files(dir="/Users/maciekswiech/Desktop/Praca/B-Droix/Ankiety CV/Kierowca kurier/CV_nieocenione_kierowca"):
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if filename.endswith('.docx'):
                print(filename)
                convert("input.docx")
        
convert_files()
