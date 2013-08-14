'''
Created on Feb 24, 2012

@author: danielfisher

“Copyright 2012 Daniel Fisher”

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import numpy
import threading
import logging

class CooccurenceMatrixTextures(object):
    '''The class performs determines the cooccurence matrix
    for a set window size over and image.  It also computes 
    and returns a number of statisitcs computed using the 
    output cooccurence matrix'''
    
    def __init__(self, image, windowRadius = 2):
        self.image = image
        self.windowRadius = windowRadius
        self.xRange = range(-self.windowRadius,self.windowRadius+1)
        self.yRange = range(-self.windowRadius,self.windowRadius+1)
        self.lock = threading.Lock()
        self.__getMatrix()
        
    def acquireLock(self):
        """Hook for multitreaded operation to acquire a lock."""
        self.lock.acquire()
        
    def releaseLock(self):
        """Hook for multithreaded operation to release a lock."""
        self.lock.release()
        
    def getDissimlarity(self):
        '''Getter method to extract the Dissimilarity'''
        return self.__dissimlarity()
    
        
    def getEntropy(self): 
        '''Getter method to extract the Entropy'''
        return self.__entropy()
        
        
    def getASM(self):
        '''Getter method to extract the Angular Second Momentum'''
        return self.__asm() 
        
    
    def getMean(self):
        '''Getter method to extract the Mean'''
        return self.__mean()
        
    
    def getVarMean(self):
        '''Getter method to obtain the Mean and Variance'''
        #get texture
        mean = self.__mean()
        var = self.__variance(mean)
        return var, mean
        
    def getCorrVarMean(self):
        '''Getter method to obtain the Mean, Variance and Correlation'''
        #get texture
        mean = self.__mean()
        var = self.__variance(mean)
        corr = self.__correlation(mean, var)
        return corr, var, mean
                     
    def __dissimlarity(self):
        '''This method gets the contrast image derived from the GLCM'''
        
        #first create the contrast matrix
        topHalf = self.__diagMatrix()
        weights = topHalf + topHalf.T
        print weights
        
        #flatten
        weights = weights.flatten()
         
        #apply to the image through broadcasting 
        weightedGLCM = self.GLCM * weights 
        
        #sum along the third axis
        dissimilarity = numpy.sum(weightedGLCM,2)
        
        #return the result
        return dissimilarity
        
    def __entropy(self): 
        '''This method gets the entropy from the GLCM'''
        
        #mask off the glcm
        maGLCM = numpy.ma.MaskedArray(self.GLCM, self.GLCM == 0)
        
        #calculate the logs and sum along the 3rd dimension
        entropy = numpy.sum(numpy.log(maGLCM) * maGLCM * (-1), 2)
        
        #return the result
        return entropy
    
    def __asm(self):
        '''This method returns the Angular Second Momentum for the GLCM'''
        
        return numpy.sum(self.GLCM * self.GLCM, 2)
        
    
    def __mean(self):
        '''This method gets the GLCM mean'''
        
        #create the output array
        shape = numpy.shape(self.GLCM)
        meanGLCM = numpy.zeros([shape[0],shape[1]])
       
        #Loop over each quantisation
        steps = range(0,256,16)
        coeff = 0 
        for x in steps:
            
            #sum the contents of the row and multiply by the coocurrence (GLCM Mean)
            summedStep = numpy.sum((self.GLCM[:,:,x:x+16] * coeff),2)
            coeff += 1
                
            #sum into the output image
            meanGLCM += summedStep
            
        #return the output image  
        return meanGLCM
    
    def __variance(self, mean):
        '''This method gets the GLCM variance and mean'''
        #create the output array
        shape = numpy.shape(self.GLCM)
        varGLCM = numpy.zeros([shape[0],shape[1]])
              
        #Loop over each quantisation
        steps = range(256)
        coeff = numpy.repeat(range(16), 16)
        for x in steps:
            
            #sum the contents of the row and multiply by the coocurrence (GLCM Var)
            step = self.GLCM[:,:,x] * numpy.power(coeff[x] - mean, 2)
                
            #sum into the output image
            varGLCM += step
            
        #return the output image  
        return varGLCM
                   
    def __correlation(self, mean, var):
        '''This method calucaltes the GLCM correlation, variance and mean'''
        #create the output array
        shape = numpy.shape(self.GLCM)
        corrGLCM = numpy.zeros([shape[0],shape[1]])
        
        #Loop over quantisation steps
        steps = range(256)
        coeffA = numpy.repeat(range(16), 16)
        coeffB = numpy.tile(range(16), 16) 
        for x in steps:
            
            #determine correlation (small additive value to prevent divide by zero)
            step = self.GLCM[:,:,x] * ((coeffA[x] - mean) * (coeffB[x] - mean)) / (var + 0.001)  
            
            #sum into the output image
            corrGLCM += step
            
        #return the output image  
        return corrGLCM
        
      
    def __getMatrix(self):
        '''This function computes the cooccurence matrix'''
       
        #create output arrays
        shape = numpy.shape(self.image)
        self.GLCM = numpy.zeros([shape[0],shape[1],256], numpy.int16)
        
        #set image as masked array
        maImage = numpy.ma.masked_array(self.image, self.image < 0)
        
        #scale the image to 4bit
        scaledImage = self.__scaleImage(maImage)
       
        #create the x and y indeices for the lookup
        [self.indexX, self.indexY] = numpy.meshgrid(range(shape[1]), range(shape[0]))
       
        #shift over all directions to create rotational invariance
        x = [0,1,1,1]
        y = [1,1,0,-1]
        for i in xrange(4):
           
            #roll the image
            shiftedImage = self.__roll2d(scaledImage, x[i],y[i])
           
            #compute the indices based upon the pixel values
            self.index = self.__findIndex(scaledImage, shiftedImage)
            self.indexInverse = self.__findIndex(shiftedImage, scaledImage)
            
            #do processing
            self.processElements()                    
        
        #get denominator
        den = (self.windowRadius * 2 + 1) ** 2 * 8.0
           
        #convert to probabilities
        self.GLCM = self.GLCM / den
                   
    def __scaleImage(self, image, scaledMin = 0., scaledMax = 15.):
        '''This private method scales the image to 4bit range'''
        
        #gets min and max
        imageMax = numpy.max(image)
        imageMin = numpy.min(image)
        
        #scale and replace the image
        scaledIm = (((image - imageMin) * (scaledMax-scaledMin)) / (imageMax - imageMin)) + scaledMin
        scaledIm = numpy.round(scaledIm).astype(numpy.int32)   
        #imgplot = plt.imshow(scaledIm) #@UnusedVariable
        #imgplot.set_cmap('Greys')
        #plt.show()
        return scaledIm
        
    def appendGLCM(self,i,j):
        
        #roll the indexes
        rolledIndex = self.__roll2d(self.index, i, j)
        rolledIndexInverse = self.__roll2d(self.indexInverse, i, j)
                    
        #use the indices to fill the GLCM
        self.acquireLock()
        self.GLCM[self.indexY,self.indexX,rolledIndex] += 1
        self.GLCM[self.indexY,self.indexX,rolledIndexInverse] += 1
        self.releaseLock()
    
    def processYElements(self, i):
        for j in self.yRange:
            self.appendGLCM(i,j)            
            
    def processElements(self):
        threadList = []
        for i in self.xRange:
            t = threading.Thread(name = str(i), target = self.processYElements, args = [i,])
            threadList.append(t)
            logging.info('Starting thread %d', i)
            t.start()
        logging.debug('Waiting to join threads')
        for t in threadList:
            logging.debug('About to join thread %s', t.getName())
            t.join()
           
    def __roll2d(self, image, xdir, ydir):
        '''This private method rolls the image in the given directions'''
        tmp = numpy.roll(image, ydir, 0)
        return numpy.roll(tmp, xdir, 1)  
    
    def __findIndex(self, referenceImage, neighbourImage):
        '''Finds the index into which a count must be inserted'''
        index = referenceImage * 16 + neighbourImage
        return index

    def __diagMatrix(self):
        '''Method to create a diagonal matrix'''
        
        xRange = range(16)
        yRange = range(16)
        dMatrix = numpy.zeros([16,16], numpy.int16)
        
        #Loop over the array and fill it in
        xInc = 0
        for y in yRange:
            val = 0
            for x in xRange:
                
                xPos = x + xInc
                
                if xPos >= 16:
                    continue
                else:
                    #place the value into the matrix
                    dMatrix[y,xPos] = val
                
                val+=1   
            xInc+=1
        return dMatrix
        
             
    def getGLCM(self):
        '''This returns the GLCM'''
        return self.GLCM
    
    