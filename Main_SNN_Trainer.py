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

#### NAS
nas_num_freq_channels = 64
nas_mono_stereo = 2
nas_polarity_type = 2

#### Recordings
recordings_address_size = 4
recordings_ts_tick = 1
recordings_bin_size = 20000

#### Tones
num_tones = 6

training_tones_names = ["" for i in range(num_tones)]

# Tone's names indicates their frequency (in Hz)
training_tones_names[0] = "261"
training_tones_names[1] = "349"
training_tones_names[2] = "523"
training_tones_names[3] = "698"
training_tones_names[4] = "1046"
training_tones_names[5] = "1396"

#### pyNAVIS 
pynavis_settings = MainSettings(num_channels=64, mono_stereo=1, on_off_both=1, address_size=2, ts_tick=0.2, bin_size=20000)

#### Output file
output_weights_filename = "nas2toneshidden_weights_test.csv"

#################################################################################
# Code
#################################################################################

training_tones_foldername = "training_tones"
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
training_tones_histograms = [[0.0 for i in range (nas_num_freq_channels*nas_polarity_type)] for i in range(num_tones)]

# First, for each folder
for folder_index in range (0, num_tones):
    # Go inside the folder
    current_training_tone_name = training_tones_names[folder_index]
    current_training_tone_folder_absolute_path =  abs_path_training_tones_folder + "\\" + current_training_tone_name + "\\"
    # Get a list with all the files inside the folder
    current_training_tone_filelist = os.listdir(current_training_tone_folder_absolute_path)
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
        sample_file_data = Loaders.loadAEDAT(sample_filename_absolute_path, pynavis_settings)
        sample_file_data = Functions.adapt_SpikesFile(sample_file_data, pynavis_settings)
        Functions.check_SpikesFile(sample_file_data, pynavis_settings)

        # Calculate its histogram
        sample_file_histogram = Plots.histogram(sample_file_data, pynavis_settings)
        sample_file_histogram = sample_file_histogram.astype('float64')

        # Check if it is a stereo NAS
        if(nas_mono_stereo == 2):
            # Sum both left and right histogram and calculate the mean
            for channel_index in range(0, nas_num_freq_channels*nas_polarity_type):
                # Sum
                sample_file_histogram[channel_index] += sample_file_histogram[channel_index + nas_num_freq_channels*nas_polarity_type]
                # Div by 2
                sample_file_histogram[channel_index] = float(sample_file_histogram[channel_index] / 2.0)
        
        for channel_index in range(0, nas_num_freq_channels*nas_polarity_type):
            training_tones_histograms[folder_index][channel_index] += sample_file_histogram[channel_index]
            if(file_index > 0):
                training_tones_histograms[folder_index][channel_index] = float(training_tones_histograms[folder_index][channel_index] / 2.0)

# Calculate the total number of events fired by the NAS for each tone
max_num_events_per_tone = [0.0 for i in range(0, num_tones)]

for tone_index in range(0, num_tones):
    total_activity = 0.0
    for channel_index in range(0, nas_num_freq_channels*nas_polarity_type):
        total_activity += training_tones_histograms[tone_index][channel_index]
    max_num_events_per_tone[tone_index] = total_activity

# Normalize all the values
training_tones_histograms_normalized = [[0.0 for i in range (nas_num_freq_channels*nas_polarity_type)] for i in range(num_tones)]

for dest_index in range (0, num_tones):
    for source_index in range(0, nas_num_freq_channels*nas_polarity_type):
        training_tones_histograms_normalized[dest_index][source_index] = training_tones_histograms[dest_index][source_index] / max_num_events_per_tone[dest_index]

# Finally, write a csv with the weights matrix
with open(abs_path_training_tones_folder + '\\' + output_weights_filename, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    
    for source_index in range(0, nas_num_freq_channels*nas_polarity_type):
        row_to_write = ["" for i in range(0, num_tones)]
        for dest_index in range(0, num_tones):
            row_to_write[dest_index] = str(training_tones_histograms_normalized[dest_index][source_index])
        spamwriter.writerow(row_to_write)
        