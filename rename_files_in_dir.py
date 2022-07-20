import os
def rename_file(path, category):
    files = os.listdir(path)
    for index, file in enumerate(files):
        new_filename = file[:-4] + "_" + category + ".jpg"
        # print(os.path.join(path, new_filename))
        os.rename(os.path.join(path, file), os.path.join(path, new_filename))

path  = "/Users/maciekswiech/Desktop/Praca/B-Droix/Analiza Estetyki CV/dane do analizy/kurier/kurier_przyjete/aesthetic"
category = "k_p_a"
rename_file(path, category)