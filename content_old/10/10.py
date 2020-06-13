

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




window, step = .5, .05
fpass = [0, 50]
Fs = 1000

window, step = [int(Fs*x) for x in [window, step]]
starts = range(0, train.shape[-1] - window, step)
f = mt_specpb(train[:, range(window)], NW=2)[0]
findx = (f >= fpass[0]) & (f <= fpass[1])
f = f[findx]
spectrogram = [mt_specpb(train[:, range(s, s + window)], NW=2)[1][findx] for s in starts]
