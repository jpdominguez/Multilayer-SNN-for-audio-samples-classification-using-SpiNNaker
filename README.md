# Multilayer Spiking Neural Network for audio samples classification using SpiNNaker
This is the PyNN code used in the paper titled "Multilayer Spiking Neural Network for audio samples classification using SpiNNaker", which is already accepted for publication in the 25th International Conference on Artificial Neural Networks.

In this code a 3-layer (input, hidden and output) Spiking Neural Network has been implemented in SpiNNaker for audio samples classification (8 different pure tones).


#### Abstract:
>Audio classification has always been an interesting subject of research inside the neuromorphic engineering field. Tools like Nengo or Brian, and hardware platforms like the SpiNNaker board are rapidly increasing in popularity in the neuromorphic community due to the ease of modelling spiking neural networks with them. In this manuscript a multilayer spiking neural network for audio samples classification using SpiNNaker is presented. The network consists of different leaky integrate-and-fire neuron layers. The connections between them are trained using novel firing rate based algorithms and tested using sets of pure tones with frequencies that range from 130.813 to 1396.91 Hz. The hit rate percentage values are obtained after adding a random noise signal to the original pure tone signal. The results show very good classification results (above 85% hit rate) for each class when the Signal-to-noise ratio is above 3 decibels, validating the robustness of the network configuration and the training step.
