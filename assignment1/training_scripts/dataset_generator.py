import os
import matplotlib.pyplot as plt
import requests
from matplotlib.pyplot import imshow
import matplotlib.cm as cm
import matplotlib.pylab as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFilter
import numpy as np
import random
import cv2
import os
import tensorflow as tf
import PIL


from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO


import requests
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import tempfile
import os
from faker import Faker


font_urls = [
    "http://themes.googleusercontent.com/static/fonts/oswald/v7/Y_TKV6o8WovbUd3m_X9aAA.ttf",
    "http://themes.googleusercontent.com/static/fonts/roboto/v9/zN7GBFwfMP4uA6AR0HCoLQ.ttf",
    "http://themes.googleusercontent.com/static/fonts/opensans/v6/cJZKeOuBrn4kERxqtaUH3aCWcynf_cDxXwCLxiixG1c.ttf",
    "http://themes.googleusercontent.com/static/fonts/ubuntu/v4/2Q-AW1e_taO6pHwMXcXW5w.ttf",
    "http://themes.googleusercontent.com/static/fonts/ptserif/v5/EgBlzoNBIHxNPCMwXaAhYPesZW2xOQ-xsNqO47m55DA.ttf",
    "http://themes.googleusercontent.com/static/fonts/dancingscript/v3/DK0eTGXiZjN6yA8zAEyM2S5FJMZltoAAwO2fP7iHu2o.ttf",
    "https://www.1001fonts.com/download/font/fredoka-one.one-regular.ttf",
    "http://themes.googleusercontent.com/static/fonts/arimo/v5/BkZwJXYnumPMepfEA344yQ.ttf",
    "http://themes.googleusercontent.com/static/fonts/notosans/v1/LeFlHvsZjXu2c3ZRgBq9nKCWcynf_cDxXwCLxiixG1c.ttf",
    "http://themes.googleusercontent.com/static/fonts/patuaone/v3/yAXhog6uK3bd3OwBILv_SKCWcynf_cDxXwCLxiixG1c.ttf"
    ]

def generate_random_words(num_words):
    fake = Faker()
    random_words = [fake.word() for _ in range(num_words)]
    return random_words

def generate_thousand_words(num_words = 2000):
    num_words = num_words  # Number of random words to generate
    random_words = generate_random_words(num_words)
    return random_words

def download_font(url):
    # Download the font file
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Failed to download font file, {url}")
        return None

def generate_dataset(font_urls):
  for font_url in font_urls:
    font_content = download_font(font_url)
    if font_content:
      temp_font_file = tempfile.NamedTemporaryFile(delete=False)
      temp_font_file.write(font_content)
      temp_font_file.close()

      font_prop = FontProperties(fname=temp_font_file.name)
      font_list_split = font_url.split('/')
      font_name = ""
      if 'www.1001fonts.com' in font_list_split:
        print("Font:", font_list_split[-1])
        font_name = font_list_split[-1]
      else:
        print("Font:", font_list_split[-3])
        font_name = font_list_split[-3]

      random_words = generate_thousand_words(num_words = 2000)
      for index, word in enumerate(random_words):
        text = word

        text_width = len(text) * 0.6  # Estimated width of text based on characters
        plt.figure(figsize=(text_width, 1))
        plt.axis('off')  # Turn off
        plt.tight_layout(pad=0)

        # Plot the text using the specified font
        plt.text(0.5, 0.5, text, fontproperties=font_prop, fontsize=50, ha='center')
        plt.savefig(f"/content/dataset/{font_name}-{word}-{index}.png", bbox_inches = "tight")
