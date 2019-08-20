import numpy as np
from os import path

from pygenn.genn_model import (create_custom_neuron_class,
                               create_custom_current_source_class,
                               GeNNModel)
from pygenn.genn_wrapper import NO_DELAY

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
IF_PARAMS = {"Vthr": 5.0}
TIMESTEP = 1.0
PRESENT_TIMESTEPS = 100
INPUT_CURRENT_SCALE = 1.0 / 100.0

# ----------------------------------------------------------------------------
# Custom GeNN models
# ----------------------------------------------------------------------------
# Very simple integrate-and-fire neuron model
if_model = create_custom_neuron_class(
    "if_model",
    param_names=["Vthr"],
    var_name_types=[("V", "scalar"), ("SpikeCount", "unsigned int")],
    sim_code="$(V) += $(Isyn) * DT;",
    reset_code="""
    $(V) = 0.0;
    $(SpikeCount)++;
    """,
    threshold_condition_code="$(V) >= $(Vthr)")

# Current source model which injects current with a magnitude specified by a state variable
cs_model = create_custom_current_source_class(
    "cs_model",
    var_name_types=[("magnitude", "scalar")],
    injection_code="$(injectCurrent, $(magnitude));")

# ----------------------------------------------------------------------------
# Build model
# ----------------------------------------------------------------------------
# Create GeNN model
model = GeNNModel("float", "tutorial_1")
model.dT = TIMESTEP

# Load weights
weights = []
while True:
    filename = "weights_%u_%u.npy" % (len(weights), len(weights) + 1)
    if path.exists(filename):
        weights.append(np.load(filename))
    else:
        break

# Initial values to initialise all neurons to
if_init = {"V": 0.0, "SpikeCount":0}

# Create first neuron layer
neuron_layers = [model.add_neuron_population("neuron0", weights[0].shape[0],
                                             if_model, IF_PARAMS, if_init)]

# Create subsequent neuron layer
for i, w in enumerate(weights):
    neuron_layers.append(model.add_neuron_population("neuron%u" % (i + 1),
                                                     w.shape[1], if_model,
                                                     IF_PARAMS, if_init))

# Create synaptic connections between layers
for i, (pre, post, w) in enumerate(zip(neuron_layers[:-1], neuron_layers[1:], weights)):
    model.add_synapse_population(
        "synapse%u" % i, "DENSE_INDIVIDUALG", NO_DELAY,
        pre, post,
        "StaticPulse", {}, {"g": w.flatten()}, {}, {},
        "DeltaCurr", {}, {})

# Create current source to deliver input to first layers of neurons
current_input = model.add_current_source("current_input", cs_model,
                                         "neuron0" , {}, {"magnitude": 0.0})

# Build and load our model
model.build()
model.load()

# ----------------------------------------------------------------------------
# Simulate
# ----------------------------------------------------------------------------
# Load testing data
testing_images = np.load("testing_images.npy")
testing_labels = np.load("testing_labels.npy")

# Check dimensions match network
assert testing_images.shape[1] == weights[0].shape[0]
assert np.max(testing_labels) == (weights[1].shape[1] - 1)

# Set current input by scaling first image
current_input.vars["magnitude"].view[:] = testing_images[0] * INPUT_CURRENT_SCALE

# Upload
model.push_var_to_device("current_input", "magnitude")

# Simulate
layer_spikes = [(np.empty(0), np.empty(0)) for _ in enumerate(neuron_layers)]
while model.timestep < PRESENT_TIMESTEPS:
    # Advance simulation
    model.step_time()

    # Loop through neuron layers
    for i, l in enumerate(neuron_layers):
        # Download spikes
        model.pull_current_spikes_from_device(l.name)

        # Add to data structure
        spike_times = np.ones_like(l.current_spikes) * model.t
        layer_spikes[i] = (np.hstack((layer_spikes[i][0], l.current_spikes)),
                           np.hstack((layer_spikes[i][1], spike_times)))

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Create a plot with axes for each
fig, axes = plt.subplots(len(neuron_layers), sharex=True)


# Loop through axes and their corresponding neuron populations
for a, s, l in zip(axes, layer_spikes, neuron_layers):
    # Plot spikes
    a.scatter(s[1], s[0], s=1)

    # Set title, axis labels
    a.set_title(l.name)
    a.set_ylabel("Spike number")
    a.set_xlim((0, PRESENT_TIMESTEPS * TIMESTEP))
    a.set_ylim((0, l.size))


# Add an x-axis label and translucent line showing the correct label
axes[-1].set_xlabel("Time [ms]")
axes[-1].hlines(testing_labels[0], xmin=0, xmax=PRESENT_TIMESTEPS,
                linestyle="--", color="gray", alpha=0.2)

# Show plot
plt.show()
