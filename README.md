# spaceTimeDiagramDetection

The goal of the spaceTimeDiagramDetection module is automatically analyse the 
slopes of the ridges of a given space-time diagram. In order to do so, we proceed in 3 steps:

- Using a thresholding method, we turn the greyscale input into a binary image.
- We detect the rising and falling edge of each ridge. We then merge them together.
- We apply linear regression alogrithms in order to compute the slope of each ridge.
    
It is noteworthy that a rising edge is detected by a black pixel followed by a sequence of $n$ white pixels.
Conversely, a falling edge is caracterized by a white pixel followed by a sequence of $n$ black pixels.

For more info, one can refer to the <a href="https://github.com/Lucas-Reding/spaceTimeDiagramDetection/blob/main/manual.pdf">pdf file</a> describing the algorithm in greater detail.

## How to use the module

In order to use this module, one can either

- Create their own script using the provided function as well as the documentation.
- Use the provided Python script <a href="https://github.com/Lucas-Reding/spaceTimeDiagramDetection/blob/main/spaceTimeAnalysis.py">spaceTimeAnalysis</a>. 

## The Python Script

Usage: 
    
    
spaceTimeAnalysis [-h] [-o OUTPUT] [-c CONFIG] [-a APPROX] [-e EPSILON] [-n N] [-N NOISE] [-t THRESHOLD]
                         [-r RANGE RANGE] [-w WIDTH] [-s SLOPE] [-R RANGE] [-v]
                         input

positional arguments:
  input                 Path to the file to analyse.

options:
    
    -h, --help            show this help message and exit
    -o OUTPUT, --output OUTPUT             path to the output file.
    -c CONFIG, --config CONFIG             configuration file
    -a APPROX, --approx APPROX             approximate slope of the ridges. Default value = 1
    -e EPSILON, --epsilon EPSILON             epsilon paramater of the noneuclidean metric. Default value = 0.1
    -n N                  characteristic length of the rising and falling edge. Default value = 5
    -N NOISE, --noise NOISE             threshold value for denoising. Default value = 400
    -t THRESHOLD, --threshold THRESHOLD             threshold value used to turn the greyscale image into a binary one. Default value = 128
    -r RANGE RANGE, --range RANGE RANGE             lower and upper (in that order) bounds for the acceptable slopes. Default value = [-inf,inf]               
    -w WIDTH, --width WIDTH             characteristic width of a ridge. Default value = 100
    -s SLOPE, --slope SLOPE             slope correction coefficient. Default value = 1                    
    -R RADIUS, -- RADIUS             radius for the adjacency detection. Default value = 100
    -v, --verbose            display more information
