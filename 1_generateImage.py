# coding: utf-8

import random
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import sys
import os

# how many pictures to generate
num = 10
if len(sys.argv) > 1:
    num = int(sys.argv[1])

def genline(text, font, filename):
    '''
    generate one line
    '''
    w, h = font.getsize(text)
    image = Image.new('RGB', (w + 15, h + 15), 'white')
    brush = ImageDraw.Draw(image)
    brush.text((8, 5), text, font=font, fill=(0, 0, 0))
    image.save(filename + '.jpg')
    with open(filename + '.txt', 'w') as f:
        f.write(text)
        f.close()

if __name__ == '__main__':
    if not os.path.isdir('./lines/'):
        os.mkdir('./lines/')
    for i in range(num):
        fontname = './fonts/simkai.ttf'
        fontsize = 24
        font = ImageFont.truetype(fontname, fontsize)
        text = str(random.randint(1000000000, 9999999999))
        text = text + str(random.randint(1000000000, 9999999999))
        #text = str(random.randint(1000, 9999))
        filename = './lines/' + str(i + 1)
        genline(text, font, filename)
    pass

