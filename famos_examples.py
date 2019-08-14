import math
import numpy as np
import tempfile
from famospy import *
truep2 = 3
truep5 = 2

simDataX = np.array([i for i in range(0,10)])
simDataY = np.array([truep2**2 * x**2 - math.exp(truep5 * x) for x in simDataX])

inits = dict(p1 = 3, p2 = 4, p3 = -2, p4 = 2, p5 = 0)
defaults = dict(p1 = 0, p2 = -1, p3 = 0, p4 = -1, p5 = -1)

def cost_function(parms, binary,simX, simY):
  res = np.array([4*parms["p1"] + parms["p2"]**2 * x**2 + parms["p3"]*math.sin(x) + parms["p4"]*x - math.exp(parms["p5"] * x) for x in simX])
  diff = np.sum((res - simY)**2)
  nrPar = len([1 for i in binary if i == 1])
  nrData = len(simDataX)
  aicc = diff + 2*nrPar + 2*nrPar*(nrPar + 1)/(nrData - nrPar - 1)
  return(aicc)

def uni(low, high, size = 1):
  from numpy.random import uniform
  return(uniform(low = low, high = high, size = size))

swaps = [["p1","p5"]]

tmp = tempfile.TemporaryDirectory()
direc = "C:/Users/Meins/Desktop"

out = famos(initPar = inits,
            fitFn = cost_function,
            homedir = direc,#tmp.name,
            method = "backward",
#            doNotFit = ["p4"],
            swapParameters = swaps,
            initModelType = "mostDistant",
            verbose = True,
            simX = simDataX,
            simY = simDataY)
print(out)
#tmp.cleanup()