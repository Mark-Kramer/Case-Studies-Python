

# Set some parameters for the multitaper method (MTM)
TW = 4  # Time-bandwidth product
Fs = 1000  # Sampling frequency

# Create multitaper objects to hold the signal and method parameters
mtPlan = Multitaper(time_series=train[:, i_plan].T,  # MT for planning period
               sampling_frequency=Fs,  # ... with desired parameters
               time_halfbandwidth_product=TW)
mtMove = Multitaper(time_series=train[:, i_move].T,  # MT for movement period
               sampling_frequency=Fs,  # ... with desired parameters
               time_halfbandwidth_product=TW)

# Create connectivity objects to perform the computations
cPlan = Connectivity.from_multitaper(mtPlan)  # ... for the planning period
cMove = Connectivity.from_multitaper(mtMove)  # ... and the movement period