"""//////////////////////////////////////////////////////////////////////////////
//                                                                             //
//    Copyright  2020  Juan P. Dominguez-Morales                              //
//                                                                             //        
//    This file is part of Multilayer Spiking Neural Network for audio		   //
//    samples classification using SpiNNaker.                                  //
//                                                                             //
//    This code is free software: you can redistribute it and/or modify        //
//    it under the terms of the GNU General Public License as published by     //
//    the Free Software Foundation, either version 3 of the License, or        //
//    (at your option) any later version.                                      //
//                                                                             //
//    The code of Multilayer Spiking Neural Network for audio samples          //
//    classification using SpiNNaker is distributed in the hope that it will   //
//    be useful, but WITHOUT ANY WARRANTY; without even the implied warranty   //
//    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU       //
//    General Public License for more details.                                 //
//                                                                             //
//    You should have received a copy of the GNU General Public License        //
//    along with NAVIS Tool.  If not, see<http://www.gnu.org/licenses/> .      //
//                                                                             //
//////////////////////////////////////////////////////////////////////////////"""

#################################################################################
# Imports
#################################################################################
import os
import sys
import csv

# If pyNAVIS was not installed from pip, clone the pyNAVIS repository
# and use the following line:
#from pyNAVIS import *

# Otherwise, use this line:
import pyNAVIS

#################################################################################
# Configuration parameters
#################################################################################

#### Training flags: WARNING: only one flag should be set to "True"
is_nas2hidden_training = True
is_hidden2output_training = False

#### Recording platform flag
is_zynq = True
is_matlab = False
is_jaer = False

#### NAS layer
nas_num_freq_channels = 32  # Number of freq. channels
nas_mono_stereo = 2         # 1 = mono   ; 2 = stereo
nas_polarity_type = 2       # 1 = single ; 2 = merged

#### Hidden layer
hidden_num_tones = 5

#### Output layer
output_num_tones = 5

#### Tones
num_tones = hidden_num_tones

training_tones_names = ["" for i in range(num_tones)]

training_tones_names[0] = "261"  # Tone's names indicates their frequency (in Hz)
training_tones_names[1] = "349"
training_tones_names[2] = "523"
#training_tones_names[3] = "698"
training_tones_names[3] = "1046"
training_tones_names[4] = "1396"

#### Recordings: NOTE: don't change here!
recordings_num_channels = 0
recordings_mono_stereo = 0
recordings_on_off_both = 0
recordings_address_size = 0
recordings_ts_tick = 0.0
recordings_bin_size = 0

#### Generic source-destination assigns: NOTE: don't change here!
num_source_neurons = 0
num_dest_neurons = 0
training_tones_foldername = "training_tones\\"  # DON'T change this!
output_weights_filename = ""

# If we want to generate the weight files for the first projection
if (is_nas2hidden_training == True):
    # The number of input neurons will depend on the NAS configuration
    # Note it does not matter if the NAS is stereo, because in that 
    # case the activity of the right cochlea will be merged to
    # the activity of the left cochlea
    num_source_neurons = nas_num_freq_channels * nas_polarity_type
    num_dest_neurons = hidden_num_tones

    # The tones aedat recordings will be located in...
    training_tones_foldername += "nas2toneshidden"
    output_weights_filename = "nas2toneshidden_weights.csv"

    # NOTE: check this before running. It depends on the used
    # platform for performing the recordings
    # Check https://pypi.org/project/pyNAVIS/ for the documentation
    recordings_num_channels = nas_num_freq_channels
    recordings_mono_stereo = nas_mono_stereo - 1
    recordings_on_off_both = nas_polarity_type - 1
    recordings_address_size = 2
    if(is_zynq == True):
        recordings_ts_tick = 0.001
    else:
        recordings_ts_tick = 0.2
    recordings_bin_size = 20000
else:
    # In this case, both source and destination have the same
    # number of neurons.
    num_source_neurons = hidden_num_tones
    num_dest_neurons = output_num_tones

    training_tones_foldername += "toneshidden2tonesoutput"
    output_weights_filename = "toneshidden2tonesoutput_weights.csv"

    # Because in this case the spikes are recorded from SpiNNaker,
    # a .csv file will be used. This means no address size is needed.
    # For the ts_tick, it's needed to set to a value which produces a
    # resolution of microseconds
    recordings_num_channels = hidden_num_tones
    recordings_mono_stereo = 0
    recordings_on_off_both = 0
    recordings_address_size = 2
    recordings_ts_tick = 0.001     # Converting from ms to us
    recordings_bin_size = 20000


#### pyNAVIS 
pynavis_settings = pyNAVIS.MainSettings(num_channels=recordings_num_channels, mono_stereo=recordings_mono_stereo, on_off_both=recordings_on_off_both, address_size=recordings_address_size, ts_tick=recordings_ts_tick, bin_size=recordings_bin_size)

#################################################################################
# Code
#################################################################################

# Get the absolute path to the folder that contains the recordings
abs_path_training_tones_folder = os.path.dirname(__file__) + "\\" + training_tones_foldername

# For each tone folder:
# |---> For each file:
#    |---> Generate histogram
#    |---> If stereo NAS: split it in two histograms
#    |---> If more than 1 histogram generated
#       |---> Calculate mean histogram values
#    |---> If last histogram generated:
#       |---> Save hisotram values in a file (this is weights files)

# Create a list of list to save the histograms values
training_tones_histograms = [[0.0 for i in range (num_source_neurons)] for i in range(num_dest_neurons)]

# First, for each folder
for folder_index in range (0, num_tones):
    # Go inside the folder
    # First, we get the tone folder name
    current_training_tone_name = training_tones_names[folder_index]
    # And create the tone folder path
    current_training_tone_folder_absolute_path =  abs_path_training_tones_folder + "\\" + current_training_tone_name + "\\"

    # Get a list with all the files inside the folder
    current_training_tone_filelist = os.listdir(current_training_tone_folder_absolute_path)
    # Check how many elements there are inside
    num_files_current_training_tone_filelist = len(current_training_tone_filelist)

    print("Files found in folder " + str(current_training_tone_folder_absolute_path) + ": " )
    print(num_files_current_training_tone_filelist)
    print("List of files from folder " + str(current_training_tone_folder_absolute_path) + ": ")
    print(current_training_tone_filelist)
    
    # Read each file
    for file_index in range (0, num_files_current_training_tone_filelist):
        # Create the absolute path of the file to read
        print(file_index)
        sample_filename = current_training_tone_filelist[file_index]
        sample_filename_absolute_path = current_training_tone_folder_absolute_path + sample_filename

        # Load events file by usin pyNAVIS
        # If we are training the first projection group (nas2toneshidden), then use loadAEDAT
        # Otherwise, use loadCSV (for reading events files from SpiNNaker or others)
        if (is_nas2hidden_training == True):
            if(is_zynq == True):
                sample_file_data = pyNAVIS.Loaders.loadZynqGrabberData(sample_filename_absolute_path, pynavis_settings)
            else:
                sample_file_data = pyNAVIS.Loaders.loadAEDAT(sample_filename_absolute_path, pynavis_settings)
        else:
            sample_file_data = pyNAVIS.Loaders.loadCSV(sample_filename_absolute_path)
        
        # Then, call "pyNavis.adapt" for adapting the events' timestamps
        sample_file_data = pyNAVIS.Functions.adapt_SpikesFile(sample_file_data, pynavis_settings)
        # And check that everything is OK
        pyNAVIS.Functions.check_SpikesFile(sample_file_data, pynavis_settings)

        # Calculate its histogram
        sample_file_histogram = pyNAVIS.Plots.histogram(sample_file_data, pynavis_settings)
        # Convert the histogram type to float
        sample_file_histogram = sample_file_histogram.astype('float64')

        # When nas2toneshidden flag is active, check if it is a stereo NAS
        if((is_nas2hidden_training == True) and (nas_mono_stereo == 2)):
            # Sum both left and right histogram and calculate the mean
            for channel_index in range(0, num_source_neurons):
                # Sum
                sample_file_histogram[channel_index] += sample_file_histogram[channel_index + num_source_neurons]
                # Div by 2
                sample_file_histogram[channel_index] = float(sample_file_histogram[channel_index] / 2.0)
        
        # Finally, we calculate the mean of the previous histogram with the new histogram
        # If it is the first histogram, the mean is not calculated
        for source_index in range(0, num_source_neurons):
            training_tones_histograms[folder_index][source_index] += sample_file_histogram[source_index]
            if(file_index > 0):
                training_tones_histograms[folder_index][source_index] = float(training_tones_histograms[folder_index][source_index] / 2.0)

# Last step is to normalize the histogram for each pure tone according to the global activity
# This means, we will normalize the firing activity of each channel with the firing activity
# of the whole NAS for a especific tone.

# Calculate the total number of events fired by the NAS for each tone
max_num_events_per_tone = [0.0 for i in range(0, num_dest_neurons)]

for tone_index in range(0, num_dest_neurons):
    total_activity = 0.0
    for neuron_index in range(0, num_source_neurons):
        # Calculate the global activity of the NAS for this tone
        total_activity += training_tones_histograms[tone_index][neuron_index]
    # Save the overall activity for this tone
    max_num_events_per_tone[tone_index] = total_activity
"""for tone_index in range(0, num_dest_neurons):
    total_activity = 0.0
    for neuron_index in range(0, num_source_neurons):
        # Calculate the global activity of the NAS for this tone
        act = training_tones_histograms[tone_index][neuron_index]
        if(act > total_activity):
            total_activity = act
    # Save the overall activity for this tone
    max_num_events_per_tone[tone_index] = total_activity"""

# Normalize all the values
training_tones_histograms_normalized = [[0.0 for i in range (num_source_neurons)] for i in range(num_dest_neurons)]

for dest_index in range (0, num_dest_neurons):
    for source_index in range(0, num_source_neurons):
        training_tones_histograms_normalized[dest_index][source_index] = training_tones_histograms[dest_index][source_index] / max_num_events_per_tone[dest_index]

# Finally, write a csv with the weights matrix
with open(abs_path_training_tones_folder + '\\' + output_weights_filename, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    
    for source_index in range(0, num_source_neurons):
        row_to_write = ["" for i in range(0, num_dest_neurons)]
        for dest_index in range(0, num_dest_neurons):
            row_to_write[dest_index] = str(training_tones_histograms_normalized[dest_index][source_index])
        spamwriter.writerow(row_to_write)
        