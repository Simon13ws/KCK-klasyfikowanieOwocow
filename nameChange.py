import os
from skimage import io
from PIL import Image, ImageOps
labels = ["o","l"]#["a","b","p","o","l"]
fromPath = ["data/o/","data/l/"]# ["data/a/", "data/b/", "data/p/"]
dirPath = ["data/oranges/","data/lemons/"]# ["data/apples/","data/bananas/", "data/pears/"]

j = 0 
'''def zmniejsz(tab):
    basewidth = 30
    for i in range(0, len(tab)):
        wpercent = (basewidth / float(tab[i].size[0]))
        hsize = int((float(tab[i].size[1]) * float(wpercent)))
        tab[i] = tab[i].resize((basewidth, hsize), Image.ANTIALIAS)
    return basewidth,hsize
    '''
# ZMNIEJSZANIE
for path in fromPath:
    l = os.listdir(dirPath[j])
    i = len(l)
    for file in os.listdir(path):
        if file.endswith(".jpg") or file.endswith(".JPG"):
            original_image = Image.open(path+file)
            size = (30, 30)
            fit_and_resized_image = ImageOps.fit(original_image, size, Image.ANTIALIAS)
            fit_and_resized_image.convert('RGB').save(dirPath[j]+labels[j]+str(i)+".jpg")
            #os.rename("data/TEST/"+file,"data/TEST/"+str(i+1)+".jpg")
            i += 1
    j = j + 1