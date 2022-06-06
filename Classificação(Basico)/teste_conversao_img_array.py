from PIL import Image
from numpy import array

img = Image.open("/home/jonh/Área de Trabalho/Machine Learning/dados/flower_photos/daisy/10140303196_b88d3d6cec.jpg")

img_a = array(img)
