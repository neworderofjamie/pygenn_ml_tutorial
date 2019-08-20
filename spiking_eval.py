import numpy as np
from os import path
import matplotlib.pyplot as plt
from pygenn import genn_model, genn_wrapper

def record_current_spikes(pop, spikes, dt):
    current_spikes = pop.current_spikes
    current_spike_times = np.ones(current_spikes.shape) * dt

    if spikes is None:
        return (np.copy(current_spikes), current_spike_times)
    else:
        return (np.hstack((spikes[0], current_spikes)),
                np.hstack((spikes[1], current_spike_times)))

IF_PARAMS = {"Vthr": 5.0}
TIMESTEP = 1.0
PRESENT_TIMESTEPS = 350
INPUT_CURRENT_SCALE = 1.0 / 100.0

# Custom classes
if_model = genn_model.create_custom_neuron_class(
    "if_model",
    param_names=["Vthr"],
    var_name_types=[("V", "scalar"), ("SpikeCount", "unsigned int")],
    sim_code="$(V) += $(Isyn) * DT;",
    reset_code="""
    $(V) = 0.0;
    $(SpikeCount)++;
    """,
    threshold_condition_code="$(V) >= $(Vthr)"
)

cs_model = genn_model.create_custom_current_source_class(
    "cs_model",
    var_name_types=[("magnitude", "scalar")],
    injection_code="$(injectCurrent, $(magnitude));")

# Create GeNN model
model = genn_model.GeNNModel("float", "spiking_eval")
model.dT = TIMESTEP

# Load weights
weights = []
while True:
    filename = "weights_%u_%u.npy" % (len(weights), len(weights) + 1)
    if path.exists(filename):
        weights.append(np.load(filename))

        if len(weights) >= 2:
            assert weights[-1].shape[0] == weights[-2].shape[1]
    else:
        break

print("Loaded %u weights" % len(weights))

# Initial values to initialise all neurons to
if_init = {"V": 0.0, "SpikeCount":0}


# Create first neuron layer
neuron_layers = [model.add_neuron_population("neuron0", weights[0].shape[0], if_model, IF_PARAMS, if_init)]

# Create subsequent neuron layer
for i, w in enumerate(weights):
    neuron_layers.append(model.add_neuron_population("neuron%u" % (i + 1), w.shape[1], if_model, IF_PARAMS, if_init))

print("Created %u neuron layers" % len(neuron_layers))

# Create synaptic connections between layers
for i, (pre, post, w) in enumerate(zip(neuron_layers[:-1], neuron_layers[1:], weights)):
    model.add_synapse_population(
        "synapse%u" % i, "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
        pre, post,
        "StaticPulse", {}, {"g": w.flatten(-1)}, {}, {},
        "DeltaCurr", {}, {})

# Create current source to deliver input to first layers of neurons
current_input = model.add_current_source("current_input", cs_model, "neuron0" , {}, {"magnitude": 0.0})

# Build and load our model
model.build()
model.load()

# Load testing data
testing_images = np.load("testing_images.npy")
testing_labels = np.load("testing_labels.npy")

# Check dimensions match network
assert testing_images.shape[1] == weights[0].shape[0]
assert np.max(testing_labels) == (weights[1].shape[1] - 1)
'''
# Get views to efficiently access state variables
current_input_magnitude = current_input.vars["magnitude"].view

print(testing_labels[5])
layer_spikes = [None] * len(neuron_layers)
while model.timestep < PRESENT_TIMESTEPS:
    # If this is the first timestep of the presentation
    if model.timestep == 0:
        # Show first training image
        current_input_magnitude[:] = testing_images[5] * INPUT_CURRENT_SCALE

        # Upload
        model.push_var_to_device("current_input", "magnitude")

    # Advance simulation
    model.step_time()

    for i, l in enumerate(neuron_layers):
        model.pull_current_spikes_from_device(l.name)
        layer_spikes[i] = record_current_spikes(l, layer_spikes[i], model.t)


fig, axes = plt.subplots(len(neuron_layers), sharex=True)

for a, s in zip(axes, layer_spikes):
    a.scatter(s[1], s[0], s=1)

plt.show()
'''
# Get views to efficiently access state variables
current_input_magnitude = current_input.vars["magnitude"].view
output_spike_count = neuron_layers[-1].vars["SpikeCount"].view
layer_voltages = [l.vars["V"].view for l in neuron_layers]

while model.timestep < (PRESENT_TIMESTEPS * 1000):
    # Calculate the timestep within the presentation
    timestep_in_example = model.timestep % PRESENT_TIMESTEPS
    example = int(model.timestep // PRESENT_TIMESTEPS)

    # If this is the first timestep of the presentation
    if timestep_in_example == 0:
        current_input_magnitude[:] = testing_images[example] * INPUT_CURRENT_SCALE
        model.push_var_to_device("current_input", "magnitude")

        for l, v in zip(neuron_layers, layer_voltages):
            v[:] = 0.0
            model.push_var_to_device(l.name, "V")

        # Zero spike count
        output_spike_count[:] = 0
        model.push_var_to_device(neuron_layers[-1].name, "SpikeCount")

    # Advance simulation
    model.step_time()

    if timestep_in_example == (PRESENT_TIMESTEPS - 1):
        model.pull_var_from_device(neuron_layers[-1].name, "SpikeCount")

        most_spikes = np.argmax(output_spike_count)
        if most_spikes == testing_labels[example]:
            print("CORRECT!")
        else:
            print("INCORRECT!")
