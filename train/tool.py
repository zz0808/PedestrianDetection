# import PIL
from PIL import Image
import os 


def main(path):

	files = os.listdir(path)

	for file in files:
		# print(file)
		imageName = path + "/" + file
		image = Image.open(imageName)
		image.save(imageName)

if __name__ == '__main__':
	main("neg")
	print("ok...")