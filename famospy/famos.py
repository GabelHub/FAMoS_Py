import os

#####################################################################################################
def set_crit_parms(criticalParameters, allNames):
  
  """ Convert Critical and Swap Sets.

    Parameters:
        criticalParameters (list):A list containing lists which specify either critical parameter sets or swap parameter sets.
        allNames (list):A list containing the names of all parameters.

    Returns:
        (list): A list containing the indices of the respective critical or swap sets.   

    """
  
  if criticalParameters is None:
    return None
  out = []
  for i in range(len(criticalParameters)):
    outSub = []
    critI = criticalParameters[i]
    for j in range(len(critI)):
      outSub.append(allNames.index(critI[j]))
    
    out.append(outSub)
  
  return out

#####################################################################################################
def random_init_model(numberPar, critParms = None, doNotFit = None):
  
  """ Create a random starting model.
  
      Parameters:
          numberPar (int):The total number of parameters available.
          critParms (list):A list containing lists which specify the critical parameter sets. Needs to be given by index and not by parameter name.
          doNotFit (set): A set containing the indices of the parameters that are not supposed to be fitted.
      
      Returns:
          (set): A set containing the parameter indices of the random starting model.   
    
    """
  
  import random
  
  initMod = set()
  
  if critParms is not None:
    for i in range(len(critParms)):
      if len(critParms[i]) > 1:
        initMod = initMod | set([random.choice(critParms[i])])
      else:
        initMod = initMod | set(critParms[i])
        
  if (len(initMod) == 0) and (doNotFit is None):
    initMod = set(random.sample(range(numberPar), random.choice(range(numberPar)) + 1))
  else:
    badVals = initMod.copy()
    if doNotFit is not None:
      badVals = initMod | doNotFit
    vals = [x for x in range(numberPar) if x not in badVals]
    initMod = initMod | set(random.sample(vals, random.choice(range(len(vals))) + 1))
  
  return list(initMod)
  


#####################################################################################################
def model_appr(currentParms, criticalParms, doNotFit = None):
  
  """ Tests if a model is violating specified critical conditions.

    Parameters:
        currentParms (list):A list containing the indices of the current model.
        criticalParameters (list):A list containing lists which specify the critical parameter sets. Needs to be given by index and not by parameter name
        doNotFit (list):A list containing the indices of the parameters that are not to be fitted.

    Returns:
        (bool): True, if the model is appropriate, False, if it violates the specified critical conditions.  

    """
  
  if doNotFit is not None and any(x in currentParms for x in doNotFit):
      return False
    
  if len(criticalParms) == 0:
    return True
  
  for i in range(len(criticalParms)):
    testSet = set(criticalParms[i])
    currentSet = set(currentParms)
    if len(testSet & currentSet) == 0:
      return False
  
  return True



#####################################################################################################
def famos_directories(homedir):
  """ Creates the directories in which the results are going to be saved
  
   Parameters:
        homedir (string): The directory in which the result folders will be created

   Returns:
     Creates main directory 'FAMoS-Results', with subdirectories 'BestModel' (contains information about the best model of each run), 'Figures' (contains performance plots), 'Fits' (contains the information of all tested models) and 'TestedModels' (contains the information which models were already tested).
        
  
    """
  import os
  from pathlib import Path
  
  direc = Path(homedir) / "FAMoS-Results"
  if direc.exists() is not True:
    os.mkdir(direc)
    
  if os.path.exists(Path(direc) / "BestModel") is not True:
    os.mkdir(Path(direc) / "BestModel")
    
  if os.path.exists(Path(direc) / "Figures") is not True:
    os.mkdir(Path(direc) / "Figures")
    
  if os.path.exists(Path(direc) / "Fits") is not True:
    os.mkdir(Path(direc) / "Fits")
    
  if os.path.exists(Path(direc) / "TestedModels") is not True:
    os.mkdir(Path(direc) / "TestedModels")




#####################################################################################################
def combine_par(fitPar, allNames, defaultVal = None):
  
  """ Combine fitted and non-fitted parameters.

    Parameters:
        fitPar (dict):A dictionary containing all parameters with respective names that are supposed to be fitted
        allNames (list):A list containing the names of all parameters (fitted and non-fitted).
        defaultVal (dict):A dictionary containing the values that the non-fitted parameters should take. If NoneType, all non-fitted parameters will be set to zero. Default values can be either given by a numeric value or by the name of the corresponding parameter the value should be inherited from (NOTE: If a string is supplied, the corresponding parameter entry has to contain a numeric value). Default to None.

    Returns:
        (list): A dictionary containing the elements of fitPar and the non-fitted parameters, in the order given by allNames. The non-fitted parameters are determined by the remaining names in allNames and their values are set according to defaultVal.   

    """
  
  if not set(list(fitPar.keys())).issubset(allNames):
    raise ValueError('The names of fitPar have to be included in allNames!')
  
  if defaultVal is not None and type(defaultVal) is not dict:
    raise ValueError('defaultVal has to be either None or a dictionary')
    
  allPar = dict(zip(allNames,[0]*len(allNames)))
  for x in fitPar.keys():
    allPar[x] = fitPar[x]
    
  if defaultVal is None:
    return allPar
  else:
    for x in set(allNames).difference(set(list(fitPar.keys()))):
      if isinstance(defaultVal[x], str):
        if defaultVal[x] in fitPar.keys():
          allPar[x] = fitPar[defaultVal[x]]
        else:
          allPar[x] = defaultVal[defaultVal[x]]
      else:
        allPar[x] = defaultVal[x]
  return(allPar)
  



#####################################################################################################
def combine_and_fit(par, fitNames, parNames, fitFn, binary = None, defaultVal = None, adds = None):
  
  """ Supply combined parameters to fitting function.

    Parameters:
        par (list):A list containing the values of all parameters that are supposed to be fitted.
        fitNames (list): A list containing the names of all parameters that are supposed to be fitted. Needs to be in the same order as 'par'.
        parNames (list):A list containing the names of all parameters.
        fitFn (function):The cost or optimisation function.
        binary (list):A list containing zeroes and ones. Zero indicates that the corresponding parameter is not fitted. 
        defaultVal (dict):A dictionary containing the values that the non-fitted parameters should take. If NoneType, all non-fitted parameters will be set to zero. Default values can be either given by a numeric value or by the name of the corresponding parameter the value should be inherited from (NOTE: If a string is supplied, the corresponding parameter entry has to contain a numeric value). Default to None.
        adds (dict). A dictionary containing all necessary inputs for the cost function.

    Returns:
        (float): The corresponding value coming from the cost or optimisation function fitFn.   

    """
  
  import inspect
  #from famospy import combine_par
  
  dictpar = dict(zip(fitNames,par))

  totalPar = combine_par(fitPar = dictpar, allNames = parNames, defaultVal = defaultVal)
  funcVars = dict(parms = totalPar)
  neededVars = inspect.getfullargspec(fitFn)[0]
  neededVars.remove("parms")
  if "binary" in neededVars:
    funcVars["binary"] = binary
    neededVars.remove("binary")
  if len(neededVars) > 0:
    for i in neededVars:
      funcVars[i] = adds.get(i)

  diff = fitFn(**funcVars)
  return(diff)




#####################################################################################################
def get_model(model, homedir = os.getcwd(), allNames = None):
  """ Get the information about a fitted model.

    Parameters:
        model (str or list):Either the binary number of the model as a string (e.g. "011001"), a named vector containing the names the names of the fitted parameters or a vector containing ones and zeroes to indicate which model parameter were fitted. If the names of the fitted parameters are supplied, 'allNames' must be specified as well.
        homedir A string giving the directory in which the result folders are found.
        allNames (list):A list containing the names of all parameters (fitted and non-fitted).

    Returns:
        (list): A dictionary containing the selection criterion value of the fitted model as well as corresponding parameter values

    """
  from pathlib import Path
  from ast import literal_eval
  if isinstance(model, (str, list)) == False:
    raise TypeError("Supply a correct model definition.")
  if isinstance(model, str) == True:
    binary = model
  elif isinstance(model[0], str) == True:
    if allNames is None:
      raise TypeError("Please supply a list containing all parameter names.")
    modelBinary = [1 if i in model else 0 for i in allNames]
    binary = "".join(map(str, modelBinary))
  else:
    binary = "".join(map(str, model))
  
  modelFile = Path(homedir) / "FAMoS-Results" / "Fits" / ("Model" + binary + ".txt")
  if modelFile.exists() == False:
    raise ValueError("The specified file '" + str(modelFile) + "' does not exist.")
  else:
    with open(modelFile, "r") as f:
      res = literal_eval(f.read())
  
  return(res)



#####################################################################################################
def get_results(homedir, mrun):
  
  """ Return the results of the best model.
  
      Parameters:
          homedir (str):The working directory containing the famos files.
          critParms (list):A list containing lists which specify the critical parameter sets. Needs to be given by index and not by parameter name.
          mrun (str): The number of the famos run that is to be evaluated. Must be a three digit string in the form of '001'. Alternatively, supplying 'best' will return the best result that is found over all FAMoS runs.
      
      Returns:
          (dict): A dictionary containing the following elements:
            SCV:The value of the selection criterion of the best model.
            modelPar: The parameters of the best model.
            par:The values of the parameters corresponding to the best model.}
            binary:The binary information of the best model.
            binaryDict: The binary information in dict form.
            totalModelsTested: The total number of different models that were analysed in this run. May include repeats.
            mrun: The number of the current famos run.
            initialModel:The first model evaluated by the famos run.
    """
  import numpy
  import os
  from pathlib import Path
  from ast import literal_eval
  mrunOld = mrun
  if mrun == "best":
    bestSCV = numpy.math.inf
    oldFiles = []
    for r,d,f in os.walk(Path(homedir) / "FAMoS-Results" / "BestModel"):
      for file in f:
        oldFiles.append(file)
    
    for i in range(len(oldFiles)):
      with open(Path(homedir) / "FAMoS-Results" / "BestModel" / oldFiles[i], "r") as f:
        bm = literal_eval(f.read())
      
      if bm["SCV"] < bestSCV:
        bestSCV = bm["SCV"]
        
        mrun = oldFiles[i].replace(str("BestModel"), "")
        mrun = mrun.replace(".txt", "")

  with open(Path(homedir) / "FAMoS-Results" / "BestModel" / ("BestModel" + mrun + ".txt"), "r") as f:
    res = literal_eval(f.read())
  
  if mrunOld == "best":
    print("Best selection criterion value over all runs: " + str(round(res["SCV"], 2)))
    numberTested = 0
    testedFiles = []
    for r,d,f in os.walk(Path(homedir) / "FAMoS-Results" / "TestedModels"):
      for file in f:
        testedFiles.append(file)
    for i in range(len(testedFiles)):
      with open(Path(homedir) / "FAMoS-Results" / "TestedModels" / testedFiles[i], "r") as f:
        nt = literal_eval(f.read())
      numberTested += len(nt)
    print("Total number of tested models over all runs (might include repeats): " + str(numberTested))
  else:
    print("Best selection criterion value of run " + mrun + ": " + str(round(res["SCV"],2)))
    with open(Path(homedir) / "FAMoS-Results" / "TestedModels" / ("TestedModels" + mrun + ".txt"), "r") as f:
      numberTested = len(literal_eval(f.read()))
    print("Number of models tested during this run (might include repeats): " + str(numberTested))
  
  bestPar = res["bestPar"].copy()
  binary = res["binary"]
  #import pdb; pdb.set_trace()
  for i in reversed(range(len(binary))):
    if binary[i] == 0:
      bestPar.pop(list(bestPar.keys())[i])
      
  with open(Path(homedir) / "FAMoS-Results" / "TestedModels" / ("TestedModels" + mrun + ".txt"), "r") as f:
    initModel = literal_eval(f.read())[0]
  
  print("The parameters of the best model are: " + str(list(bestPar.keys())))
  print("Estimated parameter values: " + str(res["bestPar"]))
  print("Best model binary is: " + "".join(map(str, res["binary"])))
  output = dict(SCV = round(res["SCV"],2),
                modelPar = list(bestPar.keys()),
                par = res["bestPar"],
                binary = "".join(map(str, res["binary"])),
                binaryList = res["binary"],
                totalModelsTested = numberTested,
                mrun = mrun,
                initialModel = initModel[2:])
  return(output)



#####################################################################################################
def base_optim(binary, parms, fitFn, homedir, useOptim = True, optimRuns = 1, defaultVal = None, randomBorders = 1, conTol = 0.01, controlOptim = dict(maxiter = 1000), verbose = False, **kwargs):
  
  """ Underlying optimisation routine of famospy.

    Parameters:
        binary (list):A list containing zeroes and ones. Zero indicates that the corresponding parameter is not fitted.
        parms (dict):A dictionary containing the starting values of the optimisation procedure. Must be in the same order as 'binary'.
        fitFn (function):A cost or optimsation function. Has to take the complete parameter vector as an input argument (needs to be named 'parms') and must return must return a selection criterion value (e.g. AICc or BIC). The binary list containing the information which parameters are fitted, can also be used by taking 'binary' as an additional function input argument.
        homedir (string):A string giving the directory in which the result folders generated by famos are found.
        useOptim (bool):If True, the cost function 'fitFn' will be fitted via the minimize function from scipy.optimize. If False, the cost function will only be evaluated, meaning that users can specify their own optimisation routine inside the cost function.
        optimRuns (int):The number of times that each model will be optimised. Default to 1. Numbers larger than 1 use random initial conditions (see 'random.borders').
        defaultVal (dict):A dictionary containing the values that the non-fitted parameters should take. If None, all non-fitted parameters will be set to zero. Default values can be either given by a numeric value or by the name of the corresponding parameter the value should be inherited from (NOTE: In this case the corresponding parameter entry has to contain a numeric value). Default to None.
        randomBorders (list or function):The ranges from which the random initial parameter conditions for all optimRuns > 1 are sampled. Can be either given as a list containing the relative deviations for all parameters or as a list containing lists in which the first entry describes the lower and in the second entry describes the upper border values. Parameters are uniformly sampled based on INCLUDE PYTHON FUNCTION. Default to 1 (100% deviation of all parameters). Alternatively, functions such as INCLUDE PYTHON FUNCTION, INCLUDE PYTHON FUNCTION, etc. can be used if the additional arguments are passed along as well.
        conTol (float):The relative convergence tolerance. If useOptim is True, famos will rerun the optimisation routin until the relative improvement between the current and the last fit is less than conTol. Default is set to 0.01, meaning the fitting will terminate if the improvement is less than 1% of the previous value.
        controlOptim (dict):Control parameters passed along to scipy's minimize function. For more details, see the corresponding documentation.
        verbose (bool):If True, FAMoS will output all details about the current fitting procedure.
        

    Returns:
        (file): Saves the results obtained from fitting the corresponding model parameters in the respective files, from which they can be accessed by the main function famos.   

    """
  import numpy
  from scipy.optimize import minimize
  #from famospy import combine_par, combine_and_fit
  from ast import literal_eval
  import os
  from pathlib import Path
  #get the indices of the fitted and the not-fitted parameter
  noFitIndex = [i for i in range(len(binary)) if binary[i] == 0]
  fitIndex = [i for i in range(len(binary)) if binary[i] == 1]
  fitDict = parms.copy()
  #remove non-fitted parameters
  getAllNames = list(parms.keys())
  for i in noFitIndex:
    fitDict.pop(getAllNames[i]) 

  #number of successful runs
  k = 1
  #number of failed runs
  totalTries = 0
  
  #try k = optimRuns different combinations
  while k <= optimRuns and totalTries < (4*optimRuns):
    #set failing status to False
    abort = False
    #print number of successful runs
    if verbose:
      print("Fitting run # " + str(k))
    
    #check if the initial parameter set is working
    if k == 1:
      #take the parameter combination from the currently best model
      ranPar = fitDict.values()
      #check if the function works
      if useOptim == True:
        finiteVal = combine_and_fit(par = ranPar,
                                    fitNames = list(fitDict.keys()),
                                    parNames = getAllNames,
                                    fitFn = fitFn,
                                    defaultVal = defaultVal,
                                    binary = binary,
                                    adds = kwargs)
        works = numpy.isfinite(finiteVal)
        if works == False:
          if verbose:
            print("Inherited parameters do not work and are being skipped.")
          
          k += 1
          totalTries += 1
          continue
    else:
      #get random initial conditions and test if they work
      works = False
      tries = 0
      fitVals = numpy.array(list(fitDict.values()))
      while works == False:
        
        if callable(randomBorders):
          import inspect
          if inspect.isfunction(randomBorders) == False:
            raise TypeError("Please supply a random function for randomBorders, that can be inspected by 'inspect.getfullargspec'. This might require writing a wrapper function.")
          neededVars = set(inspect.getfullargspec(randomBorders)[0])
          randomVars = dict()
          for i in (set(kwargs.keys()) & neededVars):
            randomVars[i] = kwargs[i]
          ranPar = randomBorders(**randomVars)[fitIndex]

        elif isinstance(randomBorders, (int, float)):
          randomMin = fitVals - randomBorders*abs(fitVals)
          randomMax = fitVals + randomBorders*abs(fitVals)
          
          ranPar = numpy.random.uniform(randomMin, randomMax)
        elif isinstance(randomBorders, list):
          if isinstance(randomBorders[0], list):
            if len(randomBorders) == 1:
              randomMin = [randomBorders[0][0] for i in fitIndex]  
              randomMax = [randomBorders[0][1] for i in fitIndex]  
            else:
              randomMin = [randomBorders[i][0] for i in fitIndex]
              randomMax = [randomBorders[i][1] for i in fitIndex]
          else:
            randomMin = [fitVals[i] - randomBorders[i]*abs(fitVals[i]) for i in fitIndex]
            randomMax = [fitVals[i] + randomBorders[i]*abs(fitVals[i]) for i in fitIndex]
            
          ranPar = numpy.random.uniform(randomMin, randomMax)
        else:
          raise TypeError("random.borders must be a number, a list or a function!")
          
        if useOptim == True:
          finiteVal = combine_and_fit(par = ranPar,
                                    fitNames = list(fitDict.keys()),
                                    parNames = getAllNames,
                                    fitFn = fitFn,
                                    defaultVal = defaultVal,
                                    binary = binary,
                                    adds = kwargs)
        works = numpy.isfinite(finiteVal)
      else:
        works = True
      
      tries += 1
      if tries > 100:
        raise RuntimeError("Tried 100 times to sample valid starting conditions for the optimiser, but failed. Please check if 'randomBorders' is correctly specified.")
#         }else if(is.function(random.borders)){
#           ran.par <- R.utils::doCall(random.borders, 
#                                      args = c(list(n = length(parms)), 
#                                               list(...)))
#           ran.par <- ran.par[-no.fit]

    #specify optimisation parameters (only for the first run)
    optRun = 1
    optPrevious = 10**300
    runs = 1
    
    while True:
      #get initial parameter sets for the optimisation routine
      if runs == 1:
        optPar = list(ranPar).copy()
      else:
        optPrevious = optRun.copy()

      if useOptim == True:
        opt = minimize(combine_and_fit, 
                       x0 = optPar,

                       method='nelder-mead',
                       options={'xtol': 1e-8, 'disp': False, 'maxiter': 4000},
                       args = (list(fitDict.keys()),#fitNames
                               getAllNames,#parNames
                               fitFn,
                               binary,
                               defaultVal,
                               kwargs))
        optPar = opt.x
        optRun = opt.fun
      else:
        opt = combine_and_fit(par = optPar,
                              fitNames = list(fitDict.keys()),
                              parNames = getAllNames,
                              fitFn = fitFn,
                              defaultVal = defaultVal,
                              binary = binary,
                              adds = kwargs)
        if type(opt) is not dict:
          raise TypeError('The output of the optimisation function has to be a dictionary containing the entries  "SCV" and "parameters".')
        
        optRun = opt["SCV"]
        optPar = opt["parameters"]
        break
      
      if numpy.isfinite(optRun) == False:
        abort = True
        if verbose:
          print("Optimisation failed, run skipped.")
        
        totalTries += 1
        break
      
      if abs((optRun - optPrevious)/optPrevious) < conTol:
        break
      #update run
      runs += 1
      if verbose:
        print(optRun)
      
    #test if current run was better
    if k == 1 or optRun < optMin:
      optMin = optRun
      
      #get corresponding parameter values
      outPar = combine_par(fitPar = dict(zip(list(fitDict.keys()),optPar)),
                           allNames = getAllNames,
                           defaultVal = defaultVal)
      
      result = dict(SCV = optMin, parameters = outPar)
      #saves progress if the recent fit is the first or better than any previously saved one
      #check if this model has already been tested
      fileName = Path(homedir) / "FAMoS-Results" / "Fits" / ("Model" + "".join(map(str, binary))+ ".txt")
      if os.path.exists(fileName) is True:
        with open(fileName, "r") as f:
          resultOld = literal_eval(f.read())
        
        if resultOld["SCV"] > optMin:
          if verbose:
            print("Current fit better than previous fit. Results are overwritten.")
          
          with open(fileName, "w+") as f:
            f.write(str(result))
      else:
        with open(fileName, "w+") as f:
          f.write(str(result))
        
        if verbose:
          print("No result file present. Fitting results saved in newly created file")
          

    if abort == False or k ==1:
      k += 1
      totalTries = totalTries + 1
    
  if verbose:
    print("Fitting done.")
  return(result)
 





#####################################################################################################
def famos(initPar, fitFn, homedir = os.getcwd(), doNotFit = None, method = "forward", initModelType = "random", refit = False, useOptim = True, optimRuns = 1, defaultVal = None, swapParameters = None, criticalParameters = None, randomBorders = 1, controlOptim = dict(maxiter = 1000), conTol = 0.1, savePerformance = True, parallelise = False, logInterval = 600, verbose = False, **kwargs):
  
  """ Performs automated model selection.

    Parameters:
        initPar (dict):A dictionary containing the initial parameter names and values.
        fitFn (function):A cost or optimisation function. Has to take the complete parameter vector as an input (needs to be names 'parms') and must return a selection criterion value (e.g. AICc or BIC). In case a custom optimisation function is used, the function also needs to return the fitted parameter values.
        homedir (str):The directory in which all the results should be stored.
        doNotFit (list):A list containing the names of the parameters that are not supposed to be fitted.
        method (str):  The starting method of FAMoS. Options are "forward" (forward search), "backward" (backward elimination) and "swap" (only possible if 'critical.parameters' or 'swap.parameters' are supplied). Methods are adaptively changed over each iteration of FAMoS. Default to "forward".
        initModelType (str or list):The definition of the starting model. Options are "global" (starts with the complete model), "random" (creates a randomly sampled starting model) or "mostDistant" (uses the model most dissimilar from all other previously tested models). Alternatively, a specific model can be used by giving a list containing the corresponding names of the parameters one wants to start with. Default to "random".
         refit (bool):If True, previously tested models will be fitted again. Default to False.
         useOptim (bool):If True, the cost function fitFn will be fitted via scipy's minimize function. If False, the cost function will only be evaluated, allowing users to specify their own optimisation routine within fitFn.
         optimRuns (int):The number of times that each model will be optimised. Default to 1. Numbers larger than 1 use random initial conditions (see randomBorders).
         defaultVal (dict):A dictionary containing the values that the non-fitted parameters should take. If None, all non-fitted parameters will be set to zero. Default values can be either given by a numeric value or by the name of the corresponding parameter the value should be inherited from (NOTE: In this case the corresponding parameter entry has to contain a numeric value). Default to None.
         swapParameters (list):A list specifying which parameters are interchangeable. Each swap set is given as a list inside the list containing the names of the respective parameters. Default to None.
         criticalParameters (list):A list specifying lists of critical parameters. Critical sets are parameters sets, of which at least one parameter per set has to be present in each tested model. Default to None.
         random.borders (float, list or function):The ranges from which the random initial parameter conditions for all optimRuns larger than one are sampled. Can be either given as a float or list containing the relative deviations for all parameters or as a list containing lists in which the first entry describes the lower and the second entry describes the upper border values. Parameters are uniformly sampled based on numpy.random.uniform. Default to 1 (100\% deviation of all parameters). Alternatively, other random sampling functions can be used if the additional arguments are passed along as well (note that the use of some built-in functions requires a wrapper function).
         controlOptim (dict): Control parameters passed along to scipy's minimize function (if useOptim = True).
         conTol (float): The absolute convergence tolerance of each fitting run (see Details). Default is set to 0.1.
         savePerformance (bool):  If True, the performance of famos will be evaluated in each iteration via 'famos.performance', which will save the corresponding plots into the folder "FAMoS-Results/Figures/" (starting from iteration 3) and simultaneously show it on screen. Default to True.
         parallelise (bool): If True, famos will use parallelisation routines for model fitting.
         logInterval:The interval (in seconds) at which FAMoS informs about the current status, i.e. which parallelised models are still running and how much time has passed. Default to 600 (= 10 minutes).
         verbose (bool): If True, FAMoS will output all details about the current fitting procedure.
         
    Details:
       In each iteration, FAMoS finds all neighbouring models based on the current model and method, and subsequently tests them. If one of the tested models performs better than the current model, the model, but not the method, will be updated. Otherwise, the method, but not the model, will be adaptively changed, depending on the previously used methods.
       The cost function fitFn can take the following inputs:
         parms:A dictionary containing all parameter names and values. This input is mandatory. If useOptim = True, FAMos will automatically subset the complete parameter set into fitted and non-fitted parameters.
         binary:Optional input. The binary list contains the information which parameters are currently fitted. Fitted parameters are set to 1, non-fitted to 0. This input can be used to split the complete parameter set into fitted and non-fitted parameters if a customised optimisation function is used .
         **kwargs:Other parameters that should be passed to fitFn
       
        If useOptim = True, the cost function needs to return a single numeric value, which corresponds to the selection criterion value. However, if useOptim = False, the cost function needs to return a dictionary containing in its first entry the selection criterion value and in its second entry the dictionary containing the names and values of the fitted parameter values (non-fitted parameters are internally assessed).

    Returns:
        (dict):A dictionary containing the following elements:
          SCV: The values of the selection criterion of the best model.
          par: The values of the fitted parameters corresponding to the best model.
          binary: The binary information of the best model.
          vector: A list indicating which parameters were fitted in the best model.}
          totalModelsTested: The total number of different models that were analysed. May include repeats.
          mrun: The number of the current FAMoS run.
          initialModel: The first model evaluated by the FAMoS run.

    """
  import os
  import inspect 
  import time
  import datetime
  from pathlib import Path
  from itertools import chain
  from ast import literal_eval
  from operator import itemgetter
  from famospy import base_optim, model_appr, famos_directories, random_init_model, get_results, set_crit_parms  
  #test the appropriateness of parameters
  if isinstance(initPar, dict) is not True:
    raise TypeError("Initial parameters must be given as named vector.")
  
  if doNotFit is not None and isinstance(doNotFit, list) == False:
    raise TypeError("doNotFit must be a character vector.")
       
  if defaultVal is not None and isinstance(defaultVal, dict) == False:
    raise TypeError("defaultVal must be a dictionary.")
    
  if criticalParameters is not None and isinstance(criticalParameters, list) == False:
    raise TypeError("criticalParameters must be a list")
    
  if swapParameters is not None and isinstance(swapParameters, list) == False:
    raise TypeError("swapParameters must be a list")
  
  if method != "forward" and method != "backward" and method != "swap":
    raise TypeError("Incorrect method name! Use either 'forward', 'backward' or 'swap'.")
    
  if method == "swap" and swapParameters is None and criticalParameters is None:
    raise NameError("Please supply either a swap or a critical parameter set or change the initial search method.")
  
  if isinstance(randomBorders, (int,float,list)) == False and callable(randomBorders) == False:
    raise TypeError("randomBorders must be either a number, a list or a function.")
 
#   #reset graphical parameters on exit
#   old.par <- graphics::par("mfrow")
#   on.exit(graphics::par(mfrow = old.par))

  print("Initializing...")
  #set starting time
  start = datetime.datetime.now()
  #create FAMoS directory
  print("Create famos directory...")
  famos_directories(homedir)
  #test if a different cost function was used the last time famos was called
  infoPath = Path(homedir) / "FAMoS-Results" / "FAMoS-Info.txt"
  if infoPath.exists() == True:
    with open(infoPath, "r") as f:
      oldFunction = f.read()
    
    if str(inspect.getsourcelines(fitFn)) != oldFunction:
      userInput = "0"
      while userInput not in ["1", "2", "3"]:
        userInput = input("The previous results were generated by a different cost function and might therefore not be usable during this run. What should FAMoS do?\n 1) Continue anyway  \n 2) Delete old results and continue   \n 3) Halt\nEnter number here:")
      
      if userInput == "2":
        import shutil
        shutil.rmtree(Path(homedir) / "FAMoS-Results", ignore_errors=True)
        famos_directories(homedir)
        with open(infoPath, "w+") as f:
          f.write(str(inspect.getsourcelines(fitFn)))
      elif userInput == "3":
        import sys
        sys.exit("Execution terminated")

  else:
    with open(infoPath, "w+") as f:
      f.write(str(inspect.getsourcelines(fitFn)))
      
  #get mrun for unique labelling of this run
  oldFiles = []
  for r,d,f in os.walk(Path(homedir) / "FAMoS-Results" / "TestedModels"):
    for file in f:
      file = file.replace(".txt","")
      file = file.replace("TestedModels", "")
      oldFiles.append(int(file))
  
  if len(oldFiles) == 0:
    mrun = str(1)
  else:
    mrun = str(max(oldFiles) + 1)
  
  if len(mrun) < 3:
    for i in range(0,3 - len(mrun)):
      mrun = "0" + mrun
   
  print("Algorithm run: " + mrun)
  
  # get all parameter names
  allNames = list(initPar.keys())
  
  #prepare critical and swap parameters for algorithm
  if criticalParameters is None:
    noCrit = True
    critParms = list()
  else:
    noCrit = False
    critParms = set_crit_parms(criticalParameters = criticalParameters, allNames = allNames)
    
  if swapParameters is None:
    noSwap = True
    swapParms = list()
  else:
    noSwap = False
    swapParms = set_crit_parms(criticalParameters = swapParameters, allNames = allNames)
    
  #get indices of do.not.fit
  if doNotFit is not None:
    doNotFit = [i for i in range(len(allNames)) if allNames[i] in doNotFit]

  if critParms is not None and doNotFit is not None and len(set(chain.from_iterable(critParms)) & set(doNotFit)) > 0:
    raise ValueError("The critical set contains parameters that are not supposed to be fitted. Please change either 'criticalParameters' or 'doNotFit'!")
 
  #set starting model
  if isinstance(initModelType, list):
    #take the model specified by the user
    initModel = [i for i in range(len(allNames)) if allNames[i] in initModelType]
    #check if model is appropriate
    ma = model_appr(currentParms = initModel,
                    criticalParms = critParms,
                    doNotFit = doNotFit)
    if ma == False:
      raise ValueError("The specified initial model violates critical conditions or the doNotFit specifications.")
  elif isinstance(initModelType, str):
    if initModelType == "global":
      #use the global model
      initModel = [i for i in range(len(allNames)) if i not in doNotFit]
    elif initModelType == "random":
      #set a random starting model
      initModel = random_init_model(numberPar = len(allNames),
                                  critParms = critParms,
                                  doNotFit = set(doNotFit))
    elif initModelType == "mostDistant":
      print("Not working")
      #             "most.distant" = {
#               #check if previous files exist
#               filenames <- list.files(paste0(homedir,"/FAMoS-Results/TestedModels/"),
#                                       pattern="*.rds",
#                                       full.names=TRUE)
#               if(length(filenames) == 0){
#                 stop("No previously tested models available. Please use another option for init.model.type.")
#               }
#               #get a random initial model
#               init.model <- which(get.most.distant(input = homedir)[[3]] == 1)
#             }
    else:
      raise ValueError("Please use either 'global', 'random' or 'mostDistant' as initModelType. Alternatively, specify a list of parameter names.")
    
  else:
    raise TypeError("initModelType must be a string or a string list.")
 
  #set initial parameters as starting parameters
  bestPar = initPar
 
  #define count variable for runs in total (counting how many different methods were tested)
  modelRun = 1
  #create storage for tested models
  modelsTested = list()
  modelsPerRun = list()
  saveSCV = list()
  
  if refit == True:
    print("Refitting enabled.")
  else:
    print("Refitting disables.")
  
  print("Starting algorithm with method '" + method + "'.")

  #########################
  ##### MAIN ROUTINE ######
  #########################
  while True:
    #define model to be used in the first run
    if modelRun == 1:
      print("\nfamos iteration # " + str(modelRun) + " - fitting starting model.")
      
      pickModel = initModel.copy()
      #save model for next step
      pickModelPrev = pickModel.copy()
      #mark the fitted parameters
      currModel = [1 if i in initModel else 0 for i in range(len(allNames))]
      #currModelNames = [allNames[i] for i in range(len(allNames)) if i in initModel]
      currModelAll = [currModel]
      #set history
      previous = method
    else:
      print("\nfamos iteration # " + str(modelRun) + " - method: " + method)
      if method == "forward":
        #get parameters that are currently not in the model
        parmsLeft = [i for i in range(len(bmBinary)) if bmBinary[i] == 0]
        
        #check if the current model is the global model
        failedGlobal = False
        if len(parmsLeft) == 0:
          if verbose:
            print("No addable parameters left. Switch to backward elimination.")
          previous = "forward"
          method = "backward"
          failedGlobal = True
          continue
        #create all neighbouring models based on forward search
        currModelAll = []
        for j in range(len(parmsLeft)):
          if verbose:
            print("Add parameter " + allNames[parmsLeft[j]])
          
          #get the indices of the currently tested parameter set
          pickModel = pickModelPrev + [parmsLeft[j]]
          #transform to binary information
          currModel = [1 if i in pickModel else 0 for i in range(len(allNames))]
          
          #test if model violates the critical conditions
          ma =  model_appr(currentParms = pickModel,
                    criticalParms = critParms,
                    doNotFit = doNotFit)
          if ma == False:
            if verbose:
              print("Model " + "".join(map(str, currModel)) + " violates critical parameter specifications. Model skipped.")
            continue
          
          
          #check if model has been tested before
          if currModel in modelsTested and refit == False:
            if verbose:
              print("Model has already been tested. Model skipped.")
            continue
          
          #if model was neither skipped nor tested before, add to the testing catalogue
          currModelAll.append(currModel)
      elif method == "backward":
        #get parameters that are currently in the model
        parmsLeft = pickModelPrev.copy()
        #get all suitable parameter combinations
        if len(parmsLeft) == 1:
          if verbose:
            print("No removable parameters left. Switch to forward selection.")
          previous = "backward"
          method = "forward"
          continue
        else:
          currModelAll = []
          for j in range(len(parmsLeft)):
            if verbose:
              print("Remove parameter " + allNames[parmsLeft[j]])
            pickModel = parmsLeft.copy()
            pickModel.pop(j)
            currModel = [1 if i in pickModel else 0 for i in range(len(allNames))]
            #test if model violates the critical conditions
            ma =  model_appr(currentParms = pickModel,
                  criticalParms = critParms,
                  doNotFit = doNotFit)
            if ma == False:
              if verbose:
                print("Model " + "".join(map(str, currModel)) + " violates critical parameter specifications. Model skipped.")
              continue
            #check if model has been tested before
            if currModel in modelsTested and refit == False:
              if verbose:
                print("Model has already been tested. Model skipped.")
              continue
            #if model was neither skipped nor tested before, add to the testing catalogue
            currModelAll.append(currModel)
           
          if len(currModelAll) == 0:
            previous = method
            method = "forward"
            modelRun += 1
            try:
              if failedGlobal == True:
                print("famos can neither add nor remove parameters. Algorithm terminated")
                print("Time needed: " + str(datetime.datetime.now() - start))
                finalResults = get_results(homedir, mrun)
                return(finalResults)
            except NameError:
              print("All removable parameters are critical - switch to forward search.")
              continue
            
      elif method == "swap":
        parmsComb = critParms + swapParms
        currModelAll = list()
        parmsLeft = list()
        for i in range(len(parmsComb)):
          if len(parmsComb[i]) < 1:
            continue
          
          pList = parmsComb[i]
          #save used and unused parameters of current set
          par1 = [i for i in pList if i in pickModelPrev]
          par0 = [i for i in pList if i not in par1]
          
          #create all possible combinations between used and unused parameters
          cmb = [[x,y] for x in par1 for y in par0]

          parmsLeft = cmb.copy()
 
          #create new model for each of those combinations
          if len(cmb) > 0:
            for j in range(len(cmb)):
              currModel = [1 if i in pickModel else 0 for i in range(len(allNames))]
              currModel[cmb[j][0]] = 0
              currModel[cmb[j][1]] = 1
              if verbose:
                print("Replace ", allNames[cmb[j][0]], " by ", allNames[cmb[j][1]])
            #check if model has been tested before
            if currModel in modelsTested and refit == False:
              if verbose:
                print("Model has already been tested. Model skipped.")
              continue
            
            #test if model violates the critical conditions
            pickModel = [i if currModel[i] == 1 else 0 for i in range(len(allNames))]
            #test if model violates the critical conditions
            ma =  model_appr(currentParms = pickModel,
                  criticalParms = critParms,
                  doNotFit = doNotFit)
            if ma == False:
              if verbose:
                print("Model " + "".join(map(str, currModel)) + " violates critical parameter specifications. Model skipped.")
              continue
            #if model was neither skipped nor tested before, add to the testing catalogue
            currModelAll.append(currModel)
          # if swap method fails to provide new valid models, the algorithm is being terminated
          if len(currModelAll) == 0:
            print("swap method does not yield any valid model. famos terminated.")
            print("Time needed: " + str(datetime.datetime.now() - start))
            finalResults = get_results(homedir, mrun)
            return(finalResults)

    if len(currModelAll) == 0:
      print("All neighbouring models have been tested during this run. Algorithm terminated.")
      print("Time needed: " + str(datetime.datetime.now() - start))
      finalResults = get_results(homedir, mrun)
      return(finalResults)
# 
#     #check if current model does contain parameters. If not, method is changed to forward since a model with 0 parameters would not fit our data well enough.
#     if(sum(curr.model.all) == 0) {
#       method <- "forward"
#       previous <- "backward"
#       cat("Model does not include any parameter. Continue with forward search.", sep = "\n")
#       model.run <- model.run + 1
#       models.tested <- cbind(models.tested, curr.model.all)
#       models.per.run <- cbind(models.per.run, rbind(model.run, curr.model.all))
#       save.SCV <- cbind(save.SCV, NA)
#       next
#     }
# 
    #update the catalogue of tested models
    modelsTested = modelsTested + currModelAll
    modelsPerRun = modelsPerRun + [[modelRun] + i for i in currModelAll]
    print("Time passed since start: " + str(datetime.datetime.now() - start))
    #Job submission
    if verbose:
      print("Job submission:")
    
    for j in range(len(currModelAll)):
      modelFile = Path(homedir) / "FAMoS-Results" / "Fits" / ("Model" + "".join(map(str,currModelAll[j])) + ".txt") 
      if modelFile.exists() == False or refit == True:
        if verbose:
          print("Job ID for model " + str(j + 1) + ": " + "".join(map(str,currModelAll[j])))
        if parallelise == True:
          print("not working")
        else:
          base_optim(binary = currModelAll[j],
                    parms = bestPar,
                    fitFn = fitFn,
                    homedir = homedir,
                    useOptim = useOptim,
                    optimRuns = optimRuns,
                    defaultVal = defaultVal,
                    randomBorders = randomBorders,
                    controlOptim = controlOptim,
                    conTol = conTol,
                    verbose = verbose,
                    **kwargs)
      else:
        #         assign(paste0("model",j), "no.refit")
        print("Model fit for " +  "".join(map(str,currModelAll[j])) + " exists and refitting is not enabled.")

#     if(future.off == FALSE){
#       #set looping variable
#       waiting <- TRUE
#       waited.models <- rep(1,ncol(curr.model.all))
#       time.waited <- Sys.time()
#       time.passed <- -1
#       ticker <- 0
#       ticker.time <- log.interval
# 
#       #check if the job is still running. If the job is not running anymore restart.
#       while(waiting == TRUE){
#         update.log <- FALSE
#         waiting <- FALSE
#         #cycle through all models
#         for(j in which(waited.models == 1)){
# 
#           if(future::resolved(get(paste0("model", j))) == FALSE){
#             waiting <- TRUE
#           }else{
# 
#             if(class(try(future::value(get(paste0("model", j)), std = FALSE))) == "try-error"){
#               stop(paste0("Future failed. The corresponding error message of job ",
#                           paste(curr.model.all[,j], collapse=""),
#                           " is shown above. If no output is shown, use 'future.off = TRUE' to debug."))
#             }
# 
#             #check if output was generated, including waiting period if the cluster is very busy
#             if(!file.exists(paste0(homedir,
#                                    "/FAMoS-Results/Fits/Model",
#                                    paste(curr.model.all[,j], collapse=""),
#                                    ".rds"))){
#               Sys.sleep(10)
#             }
# 
#             if(!file.exists(paste0(homedir,
#                                    "/FAMoS-Results/Fits/Model",
#                                    paste(curr.model.all[,j], collapse=""),
#                                    ".rds"))){
# 
# 
#               stop("Future is done but no output file to job ",
#                    paste(curr.model.all[,j], collapse=""),
#                    " was created. FAMoS halted.")
#             }else{
#               #update waiting variable
#               waiting <- waiting || FALSE
#               #update waiting log
#               waited.models[j] <- 0
#               update.log <- TRUE
#             }
# 
#           }
#         }
#         if(waiting == TRUE){
#           #print("Waiting ...")
#           if(time.passed == -1){
#             cat("Waiting for model fits ...", sep = "\n")
#           }
#           Sys.sleep(5)
#         }
# 
#         time.passed <- round(difftime(Sys.time(),time.waited, units = "secs")[[1]],2) - ticker*ticker.time
# 
#         #output the log for the models that is waited for (every 5 min)
#         if( (time.passed > ticker.time) ){
#           ticker <- ticker + 1
#           nr.running <-  length(which(waited.models == 1))
# 
#           timediff <- difftime(Sys.time(),time.waited, units = "secs")[[1]]
#           cat(paste0("Time spent waiting so far: ",
#                      sprintf("%02d:%02d:%02d",
#                              timediff %% 86400 %/% 3600,  # hours
#                              timediff %% 3600 %/% 60,  # minutes
#                              timediff %% 60 %/% 1), # seconds,
#                      sep = "\n"))
# 
#           if(update.log == TRUE){
#             #calculate difference in time
#             if(nr.running == ncol(curr.model.all)){
#               cat("Waiting for fits of all models ...", sep = "\n")
#             }else{
#               cat("Waiting for fits of these models:", sep = "\n")
#               cat(paste0(which(waited.models == 1)))
#               cat("",sep = "\n")
#             }
#           }
#         }
# 
#       }
#     }
    #read in files
    getSCV = []
    getPars = []
    print("Evaluate results ...")
    for j in range(len(currModelAll)):
      modelFile = Path(homedir) / "FAMoS-Results" / "Fits" / ("Model" + "".join(map(str,currModelAll[j])) + ".txt")
      if modelFile.exists() == False:
        #checking if file can be accessed
        print("Trying to read in results file of model " + "".join(map(str,currModelAll[j])))
        while True:
          time.sleep(1)
          if modelFile.exists() == True:
            break
 
      #waiting is over, read out file
      with open(modelFile, "r") as f:
        modelFit = literal_eval(f.read())
      getSCV = getSCV + [modelFit["SCV"]]
      getPars = getPars + [modelFit["parameters"]]
    #save the resulted SCVs
    saveSCV = saveSCV + getSCV
    #save SCVs with the corresponding models
    saveTestedModels = [[saveSCV[i]] + modelsPerRun[i] for i in range(len(modelsPerRun))]
    with open(Path(homedir) / "FAMoS-Results" / "TestedModels" / ("TestedModels" + mrun + ".txt"), "w+") as f:
      f.write(str(saveTestedModels))
    #if more than one model was tested, order them according to their performance
    if len(getSCV) > 1:
      indexSCV = min(enumerate(getSCV), key = itemgetter(1))[0]
    else:
      indexSCV = 0
    #print the current best fit
    currSCV = getSCV[indexSCV]
    print("Best selection criterion value of this run is " + str(round(currSCV, ndigits = 2)))

    #update if new model is better
    if modelRun == 1:
      oldSCV = currSCV
      pickModelPrev =[i for i in range(len(currModel)) if currModelAll[indexSCV][i] == 1]
      bestPar = getPars[indexSCV]
      bmBinary = currModelAll[indexSCV]
      
      with open(Path(homedir) / "FAMoS-Results" / "BestModel"/ ("BestModel" + mrun + ".txt"), "w+") as f:
        f.write(str(dict(SCV = currSCV, bestPar = bestPar, binary = bmBinary)))
    else:
# 
#       #save FAMoS performance
#       if(ncol(saveTestedModels) > 3){
# 
#         if(save.performance == T){
#           famos.performance(input = saveTestedModels,
#                             path = homedir,
#                             save.output = paste0(homedir,
#                                                  "/FAMoS-Results/Figures/Performance",
#                                                  mrun,
#                                                  ".pdf"))
#         }
# 
#         famos.performance(input = saveTestedModels,
#                           path = homedir)
# 
#       }
      if currSCV < oldSCV:
        oldSCV = currSCV
        pickModelPrev = [i for i in range(len(currModel)) if currModelAll[indexSCV][i] == 1]
        bestPar = getPars[indexSCV]
        bmBinary = currModelAll[indexSCV]
        if method == "forward":
          previous = "forward"
          method = "forward"
          print("Parameter " + allNames[parmsLeft[indexSCV]] + " was added.")
        elif method == "backward":
          previous = "backward"
          method = "backward"
          print("Parameter " + allNames[parmsLeft[indexSCV]] + " was removed.")
        elif method == "swap":
          previous = "swap"
          method = "forward"
          print("Parameter " + allNames[parmsLeft[indexSCV][0]] + " was replaced by " + allNames[parmsLeft[indexSCV][1]])
          print("Switch to forward search")
        #save best set
        with open(Path(homedir) / "FAMoS-Results" / "BestModel"/ ("BestModel" + mrun + ".txt"), "w+") as f:
          f.write(str(dict(SCV = currSCV, bestPar = bestPar, binary = bmBinary)))  
      else:
        if method == "forward":
          if previous == "forward":
            method = "backward"
          elif previous == "backward":
            if noCrit == False or noSwap == False:
              method = "swap"
            else:
              print("No better model was found. Algorithm terminated.")
              print("Time needed: " + str(datetime.datetime.now() - start))
              finalResults = get_results(homedir, mrun)
              
              #               graphics::par(mfrow = c(1,2))
#               sc.order(input = saveTestedModels,
#                        mrun = mrun)
# 
#               aicc.weights(input = saveTestedModels,
#                            mrun = mrun,
#                            reorder = TRUE)
# 
#               if(save.performance == T){
#                 sc.order(input = saveTestedModels,
#                          mrun = mrun,
#                          save.output = paste0(homedir,"/FAMoS-Results/Figures/ModelComparison",mrun,".pdf"))
# 
#                 aicc.weights(input = saveTestedModels,
#                              mrun = mrun,
#                              reorder = TRUE,
#                              save.output = paste0(homedir,"/FAMoS-Results/Figures/AkaikeWeights",mrun,".pdf"))
#               }
# 
              return(finalResults)
          elif previous == "swap":
            method = "backward"
          
          previous = "forward"
        elif method == "backward":
          if previous == "forward":
            if noCrit == False or noSwap == False:
              method = "swap"
            else:
              print("No better model was found. Algorithm terminated.")
              print("Time needed: " + str(datetime.datetime.now() - start))
              finalResults = get_results(homedir, mrun)
#               graphics::par(mfrow = c(1,2))
#               sc.order(input = saveTestedModels,
#                        mrun = mrun)
# 
#               aicc.weights(input = saveTestedModels,
#                            mrun = mrun,
#                            reorder = TRUE)
# 
#               if(save.performance == T){
#                 sc.order(input = saveTestedModels,
#                          mrun = mrun,
#                          save.output = paste0(homedir,"/FAMoS-Results/Figures/ModelComparison",mrun,".pdf"))
# 
#                 aicc.weights(input = saveTestedModels,
#                              mrun = mrun,
#                              reorder = TRUE,
#                              save.output = paste0(homedir,"/FAMoS-Results/Figures/AkaikeWeights",mrun,".pdf"))
#               }
# 
              return(finalResults)
#             }
# 
          elif previous == "backward":
            method = "forward"
          
          previous = "backward"
        elif method == "swap":
          print("No better model was found. Algorithm terminated.")
          print("Time needed: " + str(datetime.datetime.now() - start))
          finalResults = get_results(homedir, mrun)

#           # algorithm ends once swap method fails
#           cat("Best model found. Algorithm stopped.", sep = "\n")
#           final.results <- return.results(homedir, mrun)
#           timediff <- difftime(Sys.time(),start, units = "secs")[[1]]
#           cat(paste0("Time needed: ",
#                      sprintf("%02d:%02d:%02d",
#                              timediff %% 86400 %/% 3600,  # hours
#                              timediff %% 3600 %/% 60,  # minutes
#                              timediff %% 60 %/% 1), # seconds,
#                      sep = "\n"))
# 
#           graphics::par(mfrow = c(1,2))
#           sc.order(input = saveTestedModels,
#                    mrun = mrun)
# 
#           aicc.weights(input = saveTestedModels,
#                        mrun = mrun,
#                        reorder = TRUE)
# 
#           if(save.performance == T){
#             sc.order(input = saveTestedModels,
#                      mrun = mrun,
#                      save.output = paste0(homedir,"/FAMoS-Results/Figures/ModelComparison",mrun,".pdf"))
# 
#             aicc.weights(input = saveTestedModels,
#                          mrun = mrun,
#                          reorder = TRUE,
#                          save.output = paste0(homedir,"/FAMoS-Results/Figures/AkaikeWeights",mrun,".pdf"))
#           }
# 
          return(finalResults)
# 
#         }
#         )
#         cat(paste0("Switch to method '", method, "'"), sep = "\n")
#       }
#     }

    #update model run
    modelRun += 1
    print("Time passed since start: " + str(datetime.datetime.now() - start))
  
  
# =============================================================================
# import math
# import numpy as np
# import tempfile
# 
# truep2 = 3
# truep5 = 2
# 
# simDataX = np.array([i for i in range(0,10)])
# simDataY = np.array([truep2**2 * x**2 - math.exp(truep5 * x) for x in simDataX])
# 
# inits = dict(p1 = 3, p2 = 4, p3 = -2, p4 = 2, p5 = 0)
# defaults = dict(p1 = 0, p2 = -1, p3 = 0, p4 = -1, p5 = -1)
# 
# def cost_function(parms, binary,simX, simY):
#   res = np.array([4*parms["p1"] + parms["p2"]**2 * x**2 + parms["p3"]*math.sin(x) + parms["p4"]*x - math.exp(parms["p5"] * x) for x in simX])
#   diff = np.sum((res - simY)**2)
#   nrPar = len([1 for i in binary if i == 1])
#   nrData = len(simDataX)
#   aicc = diff + 2*nrPar + 2*nrPar*(nrPar + 1)/(nrData - nrPar - 1)
#   return(aicc)
# 
# def uni(low, high, size = 1):
#   from numpy.random import uniform
#   return(uniform(low = low, high = high, size = size))
#   
# swaps = [["p1","p5"]]
# 
# tmp = tempfile.TemporaryDirectory()
# direc = "C:/Users/Meins/Desktop"
# 
# out = famos(initPar = inits,
#             fitFn = cost_function,
#             homedir = direc,#tmp.name,
#             method = "swap",
#             doNotFit = ["p4"],
#             swapParameters = swaps,
#             initModelType = ["p1", "p3"],
#             verbose = True,
#             simX = simDataX,
#             simY = simDataY)
# print(out)
# #tmp.cleanup()
# =============================================================================
