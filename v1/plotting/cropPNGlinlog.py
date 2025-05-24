import sys
from glob import glob
from PIL import Image

inputfiles = glob(sys.argv[1])
inputfiles = [i for i in inputfiles if not i.endswith('_lin.png') and not i.endswith('_log.png')]

for i, inputfile in enumerate(inputfiles):
    print(f'[{i+1}/{len(inputfiles)}] {inputfile}')

    with Image.open(inputfile) as img:
        width, height = img.size

        # Left half (linear version)
        lin_crop = img.crop((0, 0, width // 2, height))
        lin_crop.save(inputfile.replace('.png', '_lin.png'))

        # Right half (log version)
        log_crop = img.crop((width // 2, 0, width, height))
        log_crop.save(inputfile.replace('.png', '_log.png'))

print("Cropping complete!")

