import os
for i in range(1,11):
    download_script = "nohup wget -c https://download.visinf.tu-darmstadt.de/data/from_games/data/{:02d}_{}.zip > logs/d{:02d}_{}.out 2>&1 &"
    os.system(download_script.format(i, "images", i, "images"))
    #os.system(download_script.format(i, "labels", i, "labels"))

