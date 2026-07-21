# NSG tool catalog — live snapshot 2026-07-21

Read directly from `https://nsgprod.sdsc.edu:8443/portal2/tools.action`. Version strings are the
portal's own. Kind is inferred from the tool name/description.

| Tool | Version | Kind | Description (portal) |
|---|---|---|---|
| AMICA on Expanse | 17 | CPU | AMICA (adaptive mixture ICA) |
| BluePyOpt on Expanse | 1.11.5 | CPU | BluePyOpt analyses |
| BluePyOpt on Expanse | 1.14.3 | CPU | BluePyOpt analyses |
| Core NEURON on Expanse | 8.2.3 | CPU | NEURON simulation |
| DEBUG BluePyOpt on Expanse | 1.1 | CPU | BluePyOpt analyses |
| EEGLAB on Expanse | Latest | CPU | EEGLAB (MATLAB); NEMAR default |
| FREESURFER on Expanse | 7.1.1 | CPU | FreeSurfer |
| Python on Expanse GPU | 3.8.10 | **GPU** | "Running TensorFlow Python models" |
| HNN on Expanse | 0.6.1 | CPU | Human Neocortical Neurosolver |
| HNN GUI on Expanse | 0.6.1 | CPU | HNN GUI |
| MATLAB on Expanse | 2020B | CPU | MATLAB |
| MRTrix (NeuroDesk) on Expanse | 3.0.3 | CPU | MRtrix via NeuroDesk container |
| NEURON on Expanse | 8.2.2 | CPU | NEURON simulation |
| NIC Converter on Expanse | 0.0.1 | CPU | NIC Converter |
| NIC Correlator on Expanse | 0.0.1 | CPU | NIC Correlator |
| NIC TDA on Expanse | 0.0.1 | CPU | NIC topological data analysis |
| **PyTorch Python on Expanse** | **2.0.1+cu117** | **GPU** | "Running PyTorch Python models" |
| HiAER-Spike FPGA-accelerated Python | 3.6.8 | FPGA | neuromorphic Python |
| Python on Expanse | 3.11.4 | CPU | BMTK, Brian 2, DEAP, HNN-Core, NetPyNE, PyNN, NEST |
| PGENESIS on Expanse | 2.4 | CPU | Parallel GENESIS |
| SpikeInterface on Expanse | 0.101.0 | CPU | SpikeInterface |
| TensorFlow Python on Expanse | 2.7.9 | GPU | TensorFlow |
| TensorFlow Python on NSG OSG | 3.7.0 | GPU (OSG) | TensorFlow on Open Science Grid |

## Notes

- **Tool IDs** (the string passed to NSG-R as `-F tool=...`) are *not* shown on this public page;
  they appear on each tool's Task-creation page after login (e.g. `NEURON_EXPANSE` for NEURON).
  For the PyTorch tool the ID is expected to be something like `PY_EXPANSE` / `PYTORCH_EXPANSE` —
  **confirm on the authenticated Task page** and record it in `nsgr/config.example.env`.
- GPU is selected by **choosing a GPU tool** (PyTorch, "Python on Expanse GPU", TensorFlow), not by a
  flag on the CPU "Python on Expanse" tool. They are distinct catalog entries.
- The presence of **MRTrix (NeuroDesk)** confirms NSG runs **containerized** tools — the route for
  bringing a custom Apptainer image with arbitrary deps.
