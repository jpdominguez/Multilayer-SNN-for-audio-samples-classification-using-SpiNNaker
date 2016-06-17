"""//////////////////////////////////////////////////////////////////////////////
//                                                                             //
//    Copyright © 2016  Juan P. Dominguez-Morales                              //
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

import pyNN.spiNNaker as p
import sys
import struct
import numpy as np
import csv

p.setup(timestep=1.0, min_delay=1.0, max_delay=16.0)

cell_params_lif = {'cm'        : 0.25, # nF
                   'i_offset'  : 0.0,
                   'tau_m'     : 20.0,
                   'tau_refrac': 2.0,
                   'tau_syn_E' : 5.0,
                   'tau_syn_I' : 5.0,
                   'v_reset'   : -70.0,
                   'v_rest'    : -65.0,
                   'v_thresh'  : -50.0
                   }
cell_params_output = {'cm'        : 0.25, # nF
                   'i_offset'  : 0.0,
                   'tau_m'     : 20.0,
                   'tau_refrac': 2.0,
                   'tau_syn_E' : 5.0,
                   'tau_syn_I' : 5.0,
                   'v_reset'   : -70.0,
                   'v_rest'    : -65.0,
                   'v_thresh'  : -55.0
                   }				   
				   
weight_to_spike = 1.0
delay = 0

aedatLength = 0
maxTimestamp = 0
minTimestamp = sys.maxsize

files = ["" for i in range(8)]
files[0] = "tones\\130.aedat"
files[1] = "tones\\174.aedat"
files[2] = "tones\\261.aedat"
files[3] = "tones\\349.aedat"
files[4] = "tones\\523.aedat"
files[5] = "tones\\698.aedat"
files[6] = "tones\\1046.aedat"
files[7] = "tones\\1396.aedat"

endTime = 0
timestampBeforeSecondFile = 0
wList = []
wList_2ndLayer = []

def createSpikeSourceArray16(aedat):
    listaResultado = [[] for i in range(128)]
    for i in range(0, len(aedat)):
		if int(aedat[i][0]/2) %2 == 0:
			listaResultado[int(aedat[i][0]/2)].append(aedat[i][1])
		else:
			listaResultado[int(aedat[i][0]/2)].append(aedat[i][1])
    return listaResultado

def loadAedat32(path):
	f = open(path, "rb")
	pos = 0
	evt = 0
	timestamp = 0
	rowID = 0
	aedatFile = []
	global minTimestamp
	global maxTimestamp
	global aedatLength
	global endTime
	try:
		while True:
			lec = f.read(1)
			if lec == "":
				break
			evt = struct.unpack("<B", lec)
			pos += 8
			evt = evt[0] << 8
			evt = evt | struct.unpack("<B", f.read(1))[0]
			pos += 8
			evt = struct.unpack("<B", f.read(1))[0]
			pos += 8
			evt <<= 8
			evt = evt | struct.unpack("<B", f.read(1))[0]
			evt &= 0x000000FF
			pos += 8

			timestamp = f.read(1)
			timestamp = struct.unpack("<B", timestamp)

			pos += 8
			timestamp = timestamp[0] << 8
			timestamp = timestamp | struct.unpack("<B", f.read(1))[0]
			pos += 8
			timestamp <<= 8
			timestamp = timestamp | struct.unpack("<B", f.read(1))[0]
			pos += 8
			timestamp <<= 8
			timestamp = (timestamp | struct.unpack("<B", f.read(1))[0])/1000
			pos += 8

			if maxTimestamp < timestamp:
				maxTimestamp = timestamp
			if rowID == 0:
				minTimestamp = timestamp

			row = np.array([evt, timestamp - minTimestamp])
			aedatFile.append(row)
			rowID += 1
		maxTimestamp = maxTimestamp - minTimestamp
		minTimestamp = 0
		endTime = maxTimestamp
		aedatLength = rowID
	finally:
		f.close()
	return aedatFile
	
def loadWeights(path):
    global wList
    with open(path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
			wList.append(row[0])

def loadWeights_2ndLayer(path):
    global wList_2ndLayer
    with open(path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
			wList_2ndLayer.append(row[0])			
				

loadWeights("synapticWeights.csv")

ssa_times = None
ssa_times = {'spike_times': createSpikeSourceArray16(loadAedat32(files[int(sys.argv[1])]))}

ssa = p.Population(64, p.SpikeSourceArray, ssa_times, label='ssa')
lif_output = p.Population(8, p.IF_curr_exp, cell_params_lif, label='lif_output')


weighted_connections = []
cont  = 0
esPos = 0
for i in range (0,63):
    for x in range(8):
		weighted_connections.append((i, x, float(float(wList[x + cont])), delay))
    cont += 16

lif_to_lif_output_proj = p.Projection(ssa, lif_output, p.FromListConnector(weighted_connections), target="excitatory")


""" SECOND LAYER"""
second_lif_layer = p.Population(8, p.IF_curr_exp, cell_params_output, label='second_lif_layer')

loadWeights_2ndLayer("synapticWeights_2ndLayer.csv")

weighted_connections_2ndLayer = []
cont  = 0
for i in range (8):
    for x in range(8):
		weighted_connections_2ndLayer.append((x, i, float(wList_2ndLayer[x + cont]), delay))
    cont += 8


lif_1_to_lif_2_proj = p.Projection(lif_output, second_lif_layer, p.FromListConnector(weighted_connections_2ndLayer), target="excitatory")


lif_output.record()
second_lif_layer.record()

p.run(endTime*2)

spikes_output_def = lif_output.getSpikes()
spikes_output_def_2nd = second_lif_layer.getSpikes()
weights = lif_to_lif_output_proj.getWeights()


neuronFirings = [0 for i in range(8)]
neuronTotal = 0
neuronFirings_2nd = [0 for i in range(8)]
neuronTotal_2nd = 0

for x in spikes_output_def:
    neuronTotal += 1
    if x[0] == 0:
        neuronFirings[0] += 1
    elif x[0] == 1:
        neuronFirings[1] += 1
    elif x[0] == 2:
        neuronFirings[2] += 1
    elif x[0] == 3:
        neuronFirings[3] += 1
    elif x[0] == 4:
        neuronFirings[4] += 1
    elif x[0] == 5:
        neuronFirings[5] += 1
    elif x[0] == 6:
        neuronFirings[6] += 1
    elif x[0] == 7:
        neuronFirings[7] += 1
		
for x in spikes_output_def_2nd:
    neuronTotal_2nd += 1
    if x[0] == 0:
        neuronFirings_2nd[0] += 1
    elif x[0] == 1:
        neuronFirings_2nd[1] += 1
    elif x[0] == 2:
        neuronFirings_2nd[2] += 1
    elif x[0] == 3:
        neuronFirings_2nd[3] += 1
    elif x[0] == 4:
        neuronFirings_2nd[4] += 1
    elif x[0] == 5:
        neuronFirings_2nd[5] += 1
    elif x[0] == 6:
        neuronFirings_2nd[6] += 1
    elif x[0] == 7:
        neuronFirings_2nd[7] += 1		

print "\n\nSPIKES FIRED FOR EACH OUTPUT NEURON - LAYER 1. FILE No. "+ str(sys.argv[1])
print "Neuron0:", neuronFirings[0]
print "Neuron1:", neuronFirings[1]
print "Neuron2:", neuronFirings[2]
print "Neuron3:", neuronFirings[3]
print "Neuron4:", neuronFirings[4]
print "Neuron5:", neuronFirings[5]
print "Neuron6:", neuronFirings[6]
print "Neuron7:", neuronFirings[7]
print "NeuronTotal", neuronTotal

print "\n\nSPIKES FIRED FOR EACH OUTPUT NEURON - LAYER 2. FILE No. "+ str(sys.argv[1])
print "Neuron0:", neuronFirings_2nd[0]
print "Neuron1:", neuronFirings_2nd[1]
print "Neuron2:", neuronFirings_2nd[2]
print "Neuron3:", neuronFirings_2nd[3]
print "Neuron4:", neuronFirings_2nd[4]
print "Neuron5:", neuronFirings_2nd[5]
print "Neuron6:", neuronFirings_2nd[6]
print "Neuron7:", neuronFirings_2nd[7]
print "NeuronTotal", neuronTotal_2nd			
