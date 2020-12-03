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
from tones import SINE_WAVE
from tones.mixer import Mixer

#################################################################################
# Configuration parameters
#################################################################################
#### Tones
num_tones = 6

training_tones_names = ["" for i in range(num_tones)]

training_tones_names[0] = "261"  # Tone's names indicates their frequency (in Hz)
training_tones_names[1] = "349"
training_tones_names[2] = "523"
training_tones_names[3] = "698"
training_tones_names[4] = "1046"
training_tones_names[5] = "1396"

#### Mixer
sample_rate = 44100 # In Hz
amplitude = 1.0
duration = 2

#### Destination folder path
abs_path_training_tones_folder = os.path.dirname(__file__) + "\\training_tones\\sounds\\"

#################################################################################
# Code
#################################################################################

# For each tone in the training tones list, create a mixer, add the tone, and
# save out the wav file

for tone_index in range(0, num_tones):
    # Create mixer, set sample rate and amplitude
    mixer = Mixer(sample_rate, amplitude)

    # Create two monophonic tracks that will play simultaneously, and set
    # initial values for note attack, decay and vibrato frequency (these can
    # be changed again at any time, see documentation for tones.Mixer
    mixer.create_track(0, SINE_WAVE, attack=0.01, decay=0.1)

    # Convert from tone_name to tone_value
    tone_frequency_value = int(training_tones_names[tone_index])
    # Add the tone to the mixer
    mixer.add_tone(0, tone_frequency_value, duration)

    # Add noise
    # TODO!

    # Create a folder for this tone
    os.mkdir(abs_path_training_tones_folder + training_tones_names[tone_index])

    # Save out the wav file
    output_wav_name = training_tones_names[tone_index] + "_mono" + ".wav"
    mixer.write_wav(abs_path_training_tones_folder + training_tones_names[tone_index] + "\\" +  output_wav_name)