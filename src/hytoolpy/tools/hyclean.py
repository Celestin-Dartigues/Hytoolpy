import numpy as np

def hyclean(t,s):
    '''HYCLEAN - Take only the values that are finite and strictly positive time
                
 Syntax: tc,sc = hyclean( t,s )
 Description:
   Take only the values that are finite and strictly positive time 
          
 Example:
   [tc,sc] = hyclean( t,s )'''
    
    condition = np.logical_and(np.isfinite(s), np.greater(s,0))

    s = np.extract(condition,s)
    t = np.extract(condition, t)
    
    return t,s
