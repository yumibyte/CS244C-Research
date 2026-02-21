# SUMMARY
This is how to generate the plots for latency and bandwidth charts from NCCL AllReduce experiments.

## ENVIRONMENT SETUP
- Refer to the setup instructions in `notes-for-farmshare.md` for the L40S 2-GPU system. This includes loading the CUDA module, setting environment variables for CUDA and NCCL, and installing micromamba to manage dependencies.

## GENERATE NCCL OUTPUT FILE
- Activate your environment:
  ```bash
  micromamba activate nccl-env
  ```
- Build the NCCL tests (without MPI for quick results):
  ```bash
  cd nccl-tests
  make MPI=0 CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"
  ```
- Run the AllReduce benchmark and save output to a file:
  ```bash
  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 | tee $MY_TEXT_FILE_NAME.txt
  ```
  - This will generate a text file (`$MY_TEXT_FILE_NAME.txt`) with the performance results.

## GENERATE BANDWIDTH PLOT
- Run the bandwidth plotting script:
  ```bash
  python plot_nccl_bw.py $MY_TEXT_FILE_NAME.txt
  ```
  - This will create bandwidth plots in the `bandwidth_graphs/` directory.

## GENERATE LATENCY PLOT
- Run the latency plotting script:
  ```bash
  python plot_nccl_latency.py $MY_TEXT_FILE_NAME.txt
  ```
  - This will create latency plots in the `latency_graphs/` directory.

## NOTES
- Both scripts will print summary statistics and save plots as PNG files.
- You can adjust the input/output filenames as needed for your experiments.