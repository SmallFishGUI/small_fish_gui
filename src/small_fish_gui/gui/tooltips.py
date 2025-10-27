
CELLPROB_TOOLTIP = """ (float) from -6. to +6.
The network predicts 3 outputs: flows in X, flows in Y, and cell “probability”. 
The predictions the network makes of the probability are the inputs to a sigmoid centered at zero (1 / (1 + e^-x)), so they vary from around -6 to +6. 
The pixels greater than the cellprob_threshold are used to run dynamics and determine ROIs. 
The default is cellprob_threshold=0.0. 
Decrease this threshold if cellpose is not returning as many ROIs as you’d expect. 
Similarly, increase this threshold if cellpose is returning too ROIs particularly from dim areas.
""" #Cellpose 4.0.6 doc

FLOW_THRESHOLD_TOOLTIP = """ (float) from 0. to 1.
The flow_threshold parameter is the maximum allowed error of the flows for each mask. 
Increase this threshold if cellpose is not returning as many ROIs as you’d expect. 
Similarly, decrease this threshold if cellpose is returning too many ill-shaped ROIs.
""" #Cellpose 4.0.6 doc