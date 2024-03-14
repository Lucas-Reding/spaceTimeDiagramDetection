"""
Usage: 
    
    
spaceTimeAnalysis [-h] [-o OUTPUT] [-c CONFIG] [-a APPROX] [-e EPSILON] [-n N] [-N NOISE] [-t THRESHOLD]
                         [-r RANGE RANGE] [-w WIDTH] [-s SLOPE] [-v]
                         input

positional arguments:
  input                 Path to the file to analyse.

options:
    
  -h, --help            show this help message and exit
  
  -o OUTPUT, --output OUTPUT
                        Path to the output file.
                                   
  -c CONFIG, --config CONFIG
                        Configuration file
                                        
  -a APPROX, --approx APPROX
                        Approximate slope of the ridges. Default value == 1
                                       
  -e EPSILON, --epsilon EPSILON
                        Epsilon paramater of the noneuclidean metric. Default value == 0.1
                                       
  -n N                  Characteristic length of the rising and falling edge. Default value == 5
  
  
  -N NOISE, --noise NOISE
                        Threshold value for denoising. Default value == 400
                        
                        
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold value used to turn the greyscale image into a binary one. Default value == 128
                        
                        
  -r RANGE RANGE, --range RANGE RANGE
                        Lower and upper (in that order) bounds for the acceptable slopes. Default value == [-inf,inf]
                        
                        
  -w WIDTH, --width WIDTH
                        Characteristic width of a ridge. Default value == 100
                        
                        
  -s SLOPE, --slope SLOPE
                        Slope correction coefficient. Default value == 1
                        
                        
  -v, --verbose         Display more information
"""

import spaceTimeDiagramDetection as stdd

#Numpy imports
from numpy import array,int32,arctan,nonzero,polyfit,where,dtype,inf
from numpy import max as max
from numpy import min as min
from numpy import sum as sum

#File managment imports
from cv2 import imread,imwrite,putText,line,cvtColor,COLOR_BGR2GRAY,FONT_HERSHEY_PLAIN
from tqdm import tqdm

#Parses the command-line arguments
import argparse
import pathlib
import sys

class DefaultAction(argparse.Action): #Detection of nondefault value
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest+'_nondefault', True)

parser = argparse.ArgumentParser("spaceTimeAnalysis", description="See the .pdf file for more in depth explanation")
parser.add_argument("input", nargs=1, help="Path to the file to analyse.", type=pathlib.Path)
parser.add_argument("-o", "--output", nargs=1, help="Path to the output file.", type=pathlib.Path, default=pathlib.Path('./out.jpg'))
parser.add_argument("-c", "--config", nargs=1, help="Configuration file", type=pathlib.Path)
parser.add_argument("-a", "--approx", help="Approximate slope of the ridges. Default value == 1", type=float, default=1, action=DefaultAction)
parser.add_argument("-e", "--epsilon", help="Epsilon paramater of the noneuclidean metric. Default value == 0.1", type=float, default=0.1, action=DefaultAction)
parser.add_argument("-n", help="Characteristic length of the rising and falling edge. Default value == 5", type=int, default=5, action=DefaultAction)
parser.add_argument("-N", "--noise", help="Threshold value for denoising. Default value == 400", type=int, default=400, action=DefaultAction)
parser.add_argument("-t", "--threshold", help="Threshold value used to turn the greyscale image into a binary one. Default value == 128", type=int, default=128, action=DefaultAction)
parser.add_argument("-r", "--range", nargs=2, help="Lower and upper (in that order) bounds for the acceptable slopes. Default value == [-inf,inf]", type=float, action=DefaultAction)
parser.add_argument("-w", "--width", help="Characteristic width of a ridge. Default value == 100", type=int, default=100, action=DefaultAction)
parser.add_argument("-s", "--slope", help="Slope correction coefficient. Default value == 1", type=float, default=1, action=DefaultAction)
parser.add_argument("-v", "--verbose", action='store_true', help="Display more information")
args = parser.parse_args()


#Sets up the parameters using the either the parsed one or the default value
if args.config is not None:
    params = {}
    with open(args.config[0].as_posix()) as conf:
        ldata = conf.readline()
        while ldata:
            s = ldata.split('=')
            if len(s) == 2:
                params[s[0]] = s[1]
            ldata = conf.readline()
    
    #Checks compatibility between the config file and the optional arguments        
    d = vars(args)
    for x in d:            
        if hasattr(args, x+'_nondefault') and d[x] != params[x]:
            raise NameError('Incompatible parameter between the config file and the optional arguments')
            sys.exit()
    
    #Sets the config filevalues
    approx = float(params['approx'])
    epislon = float(params['epsilon'])
    n = int(params['n'])
    thr_nb = int(params['noise'])
    thr_pre = int(params['threshold'])
    bounds = [float(params['r'].split(' ')[0]), float(params['r'].split(' ')[1])]
    slope = float(params['slope'])
    l = int(params['width'])
else:                
    approx = float(args.approx)
    epislon = float(args.epsilon)
    n = int(args.n)
    thr_nb = int(args.noise)
    thr_pre = int(args.threshold)
    if not args.range:
        bounds = [-inf,inf]
    else:
        bounds = [args.range[0],args.range[1]]
    slope = float(args.slope)
    l = int(args.width)
    
theta = arctan(approx)
    
#Display parameter
line_thickness = 2
font_thickness = 2

#Loads up the image and transfomrs it into greyscale
img = imread(args.input[0].as_posix()) 
pixels = array(cvtColor(img, COLOR_BGR2GRAY) , dtype=dtype(int32))

#Creates the space-time diagram object
std = stdd.spaceTimeDiagram(pixels,args.verbose)

#Detects the rising and falling edges of the ridges
std.detect_edges(thr_pre,thr_nb,n)

if args.verbose == True:
    print('Builiding and cleaning data')

std.clean_and_compute(bounds)
     
if args.verbose == True:        
    print('Merging uper and lower bounds')    

while std.merge_low_up(approx,l):
    pass

#Extract slope information from the space time diagram class
s = [] #s will contain all the detected slopes
c = [(255,255,255), (255,255,0), (255,0,255), (0,255,255), (255,0,0), (0,255,0), (0,0,255)] #colormap
j = 0
already_computed = []
s = []
k = 0

if args.verbose == True:
    print('Linear regression and display')
    ridges = tqdm(std.d.keys())
else:
    ridges = std.d.keys()

data = std.d     
for i in ridges:
    d = data[i]    
    X = []
    Y = []
    if d[2] not in already_computed:
        for j in data.keys():
            if data[j][2] == d[2]:
                mask = nonzero(std._lbls-j == 0)
                X = X + list(mask[1])
                Y = Y + list(mask[0])
        coeff,yinter = polyfit(X,Y,deg = 1)
        s.append(1/coeff*slope)
        already_computed.append(d[2])
        
        imax = where(X == max(X))[0]
        imin = where(X == min(X))[0]
        
        xmax = int(sum([X[i] for i in imax])/len(imax))
        ymax = int(xmax*coeff + yinter)
        
        xmin = int(sum([X[i] for i in imin])/len(imin))
        ymin = int(xmin*coeff + yinter)
        
        putText(img, str(k) + str(" : ") + "{:4.4f} mm/s".format(1/coeff*slope), org=(xmin+4,ymin-4), fontFace=FONT_HERSHEY_PLAIN, fontScale=1, color=c[k%len(c)],thickness=font_thickness)
        line(img, (xmin,ymin),(xmax,ymax), c[k%len(c)], thickness=line_thickness)
        k = k+1

imwrite(args.output.as_posix(),img)
print(s)
