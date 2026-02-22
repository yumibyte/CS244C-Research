## Implementation Plan: NCCL Performance Testing Across Topologies

### Overview
This plan automates running NCCL tests for all available communication topologies (e.g., ring, tree, simple, ll128, auto), collects results, and generates comparative plots for bandwidth and latency. The user specifies NUM_GPUS and other parameters via the existing run_nccl_farmshare.sh script.

---

### Requirements

- **Automated NCCL Test Runs**: Script must run NCCL tests for each topology (ring, tree, simple, ll128, auto, etc.).
- **User-Specified Parameters**: NUM_GPUS and other parameters are set by the user, not automated.
- **Result Collection**: Store results in a structured, topology-aware directory format.
- **Comparative Plotting**: Generate layered plots for bandwidth and latency, with each topology clearly color-coded and labeled.
- **New Plotting Scripts**: Create new Python files for multi-topology bandwidth and latency plotting.
- **Documentation**: Provide clear instructions for running the scripts and interpreting the plots.

---

### Implementation Steps

#### 1. Enumerate Topologies
- Identify all relevant NCCL topologies (ring, tree, simple, ll128, auto).
- Define a list of topologies to test.

#### 2. Automate NCCL Test Execution for Each Topology
- Modify or wrap run_nccl_farmshare.sh to accept a topology parameter (e.g., NCCL_ALGO or NCCL_GRAPH).
- For each topology:
  - Run NCCL tests with user-specified NUM_GPUS and parameters.
  - Save output files in a directory structure reflecting topology (e.g., results/topology_name/).

#### 3. Result Aggregation
- Standardize result filenames and directory structure for easy parsing.
- Optionally, create a manifest file listing all result files and their associated topology.

#### 4. Develop Multi-Topology Plotting Scripts
- **Bandwidth Plotting Script**:
  - New Python file (e.g., plot_multi_topology_bw.py).
  - Parse all result files.
  - Plot bandwidth curves for each topology on the same axes.
  - Assign distinct colors and line styles for each topology.
  - Add legend, labels, and title indicating topologies.
- **Latency Plotting Script**:
  - New Python file (e.g., plot_multi_topology_latency.py).
  - Similar structure to bandwidth script, but plots latency.
  - Use log scale for message size axis.
  - Ensure color coding matches bandwidth plot for consistency.

#### 5. Visualization & Usability
- Use color palettes (e.g., matplotlib’s tab10, Set2) for clear distinction.
- Add legends, axis labels, and annotations for clarity.
- Save plots in output directories, named by topology and metric.

#### 6. Documentation
- Update README or create a new guide describing:
  - How to run the test script for all topologies.
  - How to use the plotting scripts.
  - How to interpret the plots.

---

### Testing

- **Test Script Execution**: Run NCCL tests for a subset of topologies and verify result files are generated correctly.
- **Plotting Script Validation**:
  - Use sample result files to test new plotting scripts.
  - Check that all topologies are plotted, color coding is clear, and legends are accurate.
- **Edge Cases**:
  - Handle missing or malformed result files gracefully.
  - Ensure scripts work with varying numbers of topologies.
- **Visual Inspection**: Review generated plots for clarity and correctness.

---

### Assumptions & Notes

- Topology configurations are set via NCCL environment variables or command-line options.
- Result files follow a consistent format (as produced by run_nccl_farmshare.sh).
- Python plotting scripts will use matplotlib and numpy.
- Scripts will be placed in the scripts/ directory.

---

**End of Plan**