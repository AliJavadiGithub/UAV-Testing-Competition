# TGen-UQ at ICST/SBFT 2025 Tool Competition - CPS-UAV Test Case Generation Track

## Overview
This repository contains the implementation of our tool **TGen-UQ** (**T**est Case **Gen**eration for Autonomous UAV Navigation using **Q**-learning and **U**CB) for the [UAV Testing Competition](https://github.com/skhatiri/UAV-Testing-Competition). QLUCB utilizes the Q-learning algorithm to generate challenging test cases by placing obstacles along the trajectories of Unmanned Aerial Vehicles (UAVs) to test and evaluate their navigational resilience. A detailed description of the algorithm is available in our [report](reports/Task-Report.pdf).

## Getting Started

### Prerequisites
1. Follow [this guide](https://github.com/skhatiri/Aerialist#using-hosts-cli) to install the Aerialist project.
   
2. Install the Aerialist Python package:
   ```bash
   pip3 install git+https://github.com/skhatiri/Aerialist.git
   ```

3. Clone this project inside the `samples` directory of the Aerialist project:
   ```bash
   cd Aerialist/samples
   git clone https://github.com/yourusername/UAV-Testing-Competition.git
   ```

4. Create the necessary directories for logs and generated test results:
   ```bash
   cd UAV-Testing-Competition/snippets
   sudo mkdir -p logs generated_tests results/logs
   ```

### Running the Experiment
To run the experiment, execute the following command:
```bash
python3 cli.py generate case_studies/mission1.yaml 100
```

This command will generate 100 test cases based on the configuration in `mission1.yaml`.

## Team Members

* **Ali Javadi**  
   * Email: fleissig@aut.ac.ir  
   * Affiliation: Amirkabir University of Technology, Iran

* **Christian Birchler**  
   * Email: christian.birchler@{zhaw,unibe}.ch  
   * Affiliation: Zurich University of Applied Sciences & University of Bern, Switzerland  
