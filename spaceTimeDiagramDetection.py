r"""
The goal of the spaceTimeDiagramDetection module is automatically analyse the 
slopes of the ridges of a given space-time diagram. In order to do so, we proceed in 3 steps:

- Using a thresholding method, we turn the greyscale input into a binary image.
- We detect the rising and falling edge of each ridge. We then merge them together.
- We apply linear regression alogrithms in order to compute the slope of each ridge.
    
It is noteworthy that a rising edge is detected by a black pixel followed by a sequence of $n$ white pixels.
Conversely, a falling edge is caracterized by a white pixel followed by a sequence of $n$ black pixels.

For more info, one can refer to the <a href="https://github.com/Lucas-Reding/spaceTimeDiagramDetection/blob/main/manual.pdf">pdf file</a> describing the algorithm in greater detail.

#### How to use the module

In order to use this module, one can either

- Create their own script using the provided function as well as the documentation.
- Use the provided Python script <a href="https://github.com/Lucas-Reding/spaceTimeDiagramDetection/blob/main/spaceTimeAnalysis.py">spaceTimeAnalysis</a>. 

#### The Python Script

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

from cv2 import connectedComponentsWithStats
from numpy import array,sqrt,cos,sin,roll,where,delete,nonzero,zeros,int8,size
from numpy import sum as sum
from numpy import max as max
from numpy import min as min
from scipy import ndimage
from tqdm import tqdm

#from __future__ import annotations

__docformat__ = "markdown"
__version__ = "2"


class spaceTimeDiagram:
    """
    Space Time Diagram class. Represents a space-time diagram.
        
    Parameters:
        
    - img (greyscale image): greyscale image of the space time diagram.
    - verbose (boolean): if True, then displays the progress of the cleaning process.
        
    """
    
    def __init__(self,img,verbose):
        """Create a space-time diagram."""
        self.img = img
        """`img` contains the greyscale image of the space-time diagram to analyse"""
        self.verbose = verbose
        """If `True`, will display progress of different part of the process."""
        self.d = {}
        """`d` is an array containing the start and end point of each edge as well as its associated index (usefull for coloring)."""
        
        #Private attributes
        self._idx = []
        """idx (array of integers): the indexes of all the ridges."""
        self._lbls = []
        """lbls (labels image): the 2-dimensional Numpy array containing the label information of the ridges."""

    def compute_neighbors(self,r,theta,epsilon):
        r"""
        Compute the neighborhoods of characteristic radius r according to the distance `dist`. Two points are said to be neighbors if their distance 'dist' is smaller or equal to 'r'. If we define an equivalence relation $\mathcal{R}$ defined by $a \mathcal{R} b$ if and only if there exists a path of neighbors connecting $a$ and $b$. Then a neighborhood is an equivalence class for this relation. This is a discrete equivalent of the notion of connected component.
        
        Parameters:
            
        - r (positive): characteristic radius of the neighborhoods.
        - theta (between 0 and $\pi$): the angle between the horizontal axis and the principal axis of the ellipse used to define the scalar product asociated to the distance 'dist'.\n
            $\epislon$ (positive): the half-length of the secondary axis of of the ellipse used to define the scalar product asociated to the distance 'dist'.\n
            
        Returns:
            
        N (dictionary of arrays): this represents the list of all the neighborhoods found. It is encoded by a Python dictionary whose keys are representative of their respective class and whose values are the equivalence classes themselves.
        """
        N = {}
        for i in self.d.keys():
            for j in self.d.keys():
                if i!=j and dist(self.d[i][0], self.d[j][1],theta,epsilon) <= r:
                    if self.d[i][2] not in N.keys() or (self.d[i][2] in N.keys() and dist(self.d[i][0], self.d[j][1],theta,epsilon) <= dist(self.d[i][0], self.d[N[self.d[i][2]]][1],theta,epsilon)):
                        N[self.d[i][2]] = self.d[j][2]
        return N            
    
    
    def update_color(self,i,N,c):
        r"""
        Colorizes iteratively the edges according the colormap c.
        
        Parameters:
        
        - i (integer): index of the edge to color.
        - N (linked list integer): for each index i, N[i] is the next ridge index.
        - c (colormap): the list of RGB colors to iterate through.
            
        """
        self.d[i][2] = c
        if i in N.keys():
            self.update_color(N[i],N,c)
            
    def detect_edges(self,thr_pre, thr_nb, n):
        r"""
        Detects the rising and falling edge of the "dunes" and eleminates the noise.
        
        Parameters:
            
        - thr_pre (integer): lower bound for the greyscale level at which we start detecting edges.
        - thr_nb (integer): number of pixels above which a ridge is not considered noise anymore.
        - n (integer): characteristic width of an edge.
        
        Returns:
            
        The list of indexes idx as well as the labels of the noise free binary image as returned by the function connectedComponents of OpenCV2.
        """
        if self.verbose == True:
            print('Erasing the noise')
        
        bin_dw = self.img < thr_pre #We compute the rising and falling edge of each ridge
        bin_up = self.img < thr_pre
    
        pixels = self.img >= thr_pre
    
        for i in range(1,n+1):
            bin_dw = bin_dw & roll(pixels,i,axis=0)
            
        for i in range(1,n+1):
            bin_up = bin_up & roll(pixels,-i,axis=0)
            
            
        bin_up = ndimage.binary_dilation(bin_up)
        bin_dw = ndimage.binary_dilation(bin_dw)    
        binary = bin_up | bin_dw #Combining both rising and falling edge into one ridge    
    
        num,lbls,stats,centroids = connectedComponentsWithStats(int8(binary))
        self._idx = where(stats[:,4] > thr_nb) # Eliminate the noise
        self._idx = delete(self._idx[0],0)
    
        self._lbls = array(lbls)   
    
    def clean_and_compute(self,r):
        r"""
        Computes the slope of each detected edge. Removes the edges with an abnormal slope.
        
        Parameters:
            
        r (pair of real numbers): gives the lower and upper acceptability bounds for slopes.
        """
        j = 0
        h = self._lbls.shape[0]
        w = self._lbls.shape[1]
        res = zeros((h,w))
        
        
        if self.verbose == True:
            idx = tqdm(self._idx)
            
        for i in idx:
            mask = nonzero(self._lbls-i == 0)
            res = res + (self._lbls-i == 0) 
            
            imax = where(mask[0] == max(mask[0]))[0]
            l = (array([mask[1],mask[0]]).T)[imax]
            xmax,ymax = sum(l,axis=0)/size(l,axis=0)
            
            imin = where(mask[0] == min(mask[0]))[0]
            l = (array([mask[1],mask[0]]).T)[imin]
            xmin, ymin = sum(l,axis=0)/size(l,axis=0)
            
            if xmax != xmin:
                coeff = (ymax - ymin)/(xmax-xmin)
            else:
                coeff = 0
            
            if coeff > r[0] and coeff < r[1]:
                self.d[i] = [[xmax,ymax],[xmin, ymin],i]
                j = j +1
                
        self._lbls = self._lbls*res
        
    
    def merge_low_up(self,a,l):
        r"""
        Merges the lower and upper edges. This function is to be called multiple times until there is no merging left to do.
        
        Parameters:
        
        - a (real): approximate slope.
        - l (real): characteristic length between the rising edge and the falling edge.
            
        Returns:
        
        A boolean `modif` which indicates if the function has done any merging.
        """
        modif = False
        
        for i in self.d:
            p = self.d[i] 
            xmed = int((p[0][0] + p[1][0])/2)
            ymed = int((p[0][1] + p[1][1])/2)
            
            j0 = i
            m = l
            for j in self.d:
                if i != j:
                    d_comp = self.d[j]
                    x = int((d_comp[0][0] + d_comp[1][0])/2)
                    y = int((d_comp[0][1] + d_comp[1][1])/2)
                    
                    delta = ymed - (a*(xmed - x) + y)
                    if delta > 0 and delta < m:
                        m = delta
                        j0 = j
                        
            if self.d[i][2] != self.d[j0][2]:
                modif = True
            self.d[i][2] = self.d[j0][2]
        return modif

def dist(a,b,theta,epsilon):
    r"""
    Defines a noneuclidean metric. This metric serves the purpose of comparing the distance between two points in a the same ridge using an appropriate distance. $d(x,y) = \sqrt{\langle x,y\rangle}$ with $\langle x,y\rangle$ the scalar product of $x$ and $y$ associated to the ellipse $\mathcal{E}(r,\theta, \epsilon)$ defined by the semi-major axis length $r$ and its support line (itself determine by the angle $\theta$) as well as the semi-minor axis length $\epsilon r$ (see Figure \ref{ellipse}).
    <img src = "img/ellipse.png", style="width: 40vw"/>
    Note that
    $$
      \langle \left(x,y\right), \left(x,y\right)\rangle = x^2\left(\cos^2 \theta+ \frac{\sin^2 \theta}{\varepsilon^2}\right) + xy\left(\cos\theta\sin\theta-\frac{\cos\theta\sin\theta}{\varepsilon^2}\right) + y^2\left(\sin^2 \theta+ \frac{\cos^2 \theta}{\varepsilon^2}\right)
    $$
    
    Parameters:
        
    - a,b (two points on the space-time diagram): the two points we which to compute the distance of.
    - theta (between 0 and $\pi$): the angle between the horizontal axis and the principal axis of the ellipse used to define the scalar product asociated to the distance $d$.
    - $\epsilon$ (positive): the half-length of the secondary axis of of the ellipse used to define the scalar product asociated to the distance $d$.
        
    Returns:
        
    The distance between the points a and b according to the inner product associated to the ellipse defined by $\theta$ and $\epsilon$.
        
    """
    diff = array(a) - array(b)
    r = diff[0]**2*(cos(theta)**2 + sin(theta)**2/epsilon**2) + diff[0]*diff[1]*(-cos(theta) * sin(theta) / epsilon**2+cos(theta) * sin(theta)) + diff[1]**2*(sin(theta)**2 + cos(theta)**2/epsilon**2)
    return sqrt(r)



