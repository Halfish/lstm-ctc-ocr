# coding: utf-8

import random
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

num = 10000
'''
generate images in ./lines/
'''

def genline(text, font, filename):
    w, h = font.getsize(text)
    image = Image.new('RGB', (w + 10, h + 10), 'white')
    brush = ImageDraw.Draw(image)
    brush.text((5, 3), text, font=font, fill=(0, 0, 0))
    image.save(filename + '.jpg')
    with open(filename + '.txt', 'w') as f:
        f.write(text)
        f.close()
    pass

if __name__ == '__main__':
    for i in range(num):
        fontname = './fonts/simfang.ttf'
        fontsize = 24
        font = ImageFont.truetype(fontname, fontsize)
        #text = str(random.randint(1000000000, 9999999999))
        #text = text + str(random.randint(1000000000, 9999999999))
        text = str(random.randint(1000, 9999))
        filename = './lines/' + str(i + 1)
        genline(text, font, filename)
    pass

