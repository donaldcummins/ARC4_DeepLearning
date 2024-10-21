# GPU-accelerated Deep Learning on ARC4

Each of the three general-purpose GPU nodes on ARC4 is equipped with four NVIDIA V100 GPUs. The V100 was state-of-the-art in 2017 with 5120 CUDA cores, 640 tensor cores and (in the case of ARC4) 32 GB of memory. These GPUs can be used to accelerate deep-learning workflows using popular frameworks such as [PyTorch](https://pytorch.org/) and [Tensorflow](https://www.tensorflow.org/), as well as anything else that relies on NVIDIA's CUDA platform.

## Software requirements

Running deep-learning frameworks such as PyTorch on the GPU requires a supporting software stack to be installed. Key components include:

* A recent version of Python. Note that the very latest version is sometimes not supported yet.
* NVIDIA drivers. The driver version determines which version(s) of CUDA are supported.
* NVIDIA's CUDA Deep Neural Network library (cuDNN). This libary contains GPU-optimised versions of mathematical operations used in deep learning.

## Recommended installation procedure

The easiest way BY FAR to get GPU acceleration up and running on ARC4 is via the `conda` package manager. It is assumed here that you already have a working `conda` installation with the [conda-forge](https://conda-forge.org/) channel enabled. To check this, look at the `.condarc` file in your home directory. It should look like this:

```yaml
channels:
  - conda-forge
  - defaults # only if using anaconda or miniconda
channel_priority: strict
```

Make sure that `conda-forge` has a higher priority than the default Anaconda channel by listing it first (IMPORTANT).

ARC4 already has the required NVIDIA drivers. On a GPU workstation, installation would be as simple as running a `conda install` command for the relevant framework. Because the login nodes on ARC4 don't have GPUs, doing this would result in a CPU-only version of the software being installed. As a workaround, we can make use of the environment variable `CONDA_OVERRIDE_CUDA` to force installation of GPU-enabled packages (see the [ARC documentation](https://arcdocs.leeds.ac.uk/welcome.html)).

### PyTorch

To create a conda environment with PyTorch and Python 3.13, we would run:

```bash
CONDA_OVERRIDE_CUDA=12.0 conda create -n pytorch-gpu python=3.13 pytorch
```

The CONDA_OVERRIDE_CUDA environment variable indicates that we want to download the CUDA 12 build of the PyTorch package from conda-forge.

### TensorFlow

To create a conda environment with TensorFlow and Python 3.12, we would run:

```bash
CONDA_OVERRIDE_CUDA=12.0 conda create -n tensorflow-gpu python=3.12 tensorflow
```

Note that, at the time of writing, equivalent TensorFlow packages for Python 3.13 are not yet available from conda-forge. It is recommended to start by specifying the most recent Python version and work backwards until dependencies can be satisfied. Alternatively, the Python version can be omitted from the `conda create` command, however this can take longer as it may perform a search across all available Python versions.

### GPU vs CPU versions

Before installing the requested packages, conda will show a list of what is to be downloaded. Here is example output from the PyTorch command above:

```
The following NEW packages will be INSTALLED:

  _libgcc_mutex      conda-forge/linux-64::_libgcc_mutex-0.1-conda_forge
  _openmp_mutex      conda-forge/linux-64::_openmp_mutex-4.5-2_kmp_llvm
  bzip2              conda-forge/linux-64::bzip2-1.0.8-h4bc722e_7
  ca-certificates    conda-forge/linux-64::ca-certificates-2024.8.30-hbcca054_0
  cpython            conda-forge/noarch::cpython-3.13.0-py313hd8ed1ab_100
  cuda-cudart        conda-forge/linux-64::cuda-cudart-12.6.77-h5888daf_0
  cuda-cudart_linux~ conda-forge/noarch::cuda-cudart_linux-64-12.6.77-h3f2d84a_0
  cuda-nvrtc         conda-forge/linux-64::cuda-nvrtc-12.6.77-hbd13f7d_0
  cuda-nvtx          conda-forge/linux-64::cuda-nvtx-12.6.77-hbd13f7d_0
  cuda-version       conda-forge/noarch::cuda-version-12.6-h7480c83_3
  cudnn              conda-forge/linux-64::cudnn-9.3.0.75-h93bb076_0
  filelock           conda-forge/noarch::filelock-3.16.1-pyhd8ed1ab_0
  fsspec             conda-forge/noarch::fsspec-2024.9.0-pyhff2d567_0
  gmp                conda-forge/linux-64::gmp-6.3.0-hac33072_2
  gmpy2              conda-forge/linux-64::gmpy2-2.1.5-py313h11186cd_2
  icu                conda-forge/linux-64::icu-75.1-he02047a_0
  jinja2             conda-forge/noarch::jinja2-3.1.4-pyhd8ed1ab_0
  ld_impl_linux-64   conda-forge/linux-64::ld_impl_linux-64-2.43-h712a8e2_1
  libabseil          conda-forge/linux-64::libabseil-20240722.0-cxx17_h5888daf_1
  libblas            conda-forge/linux-64::libblas-3.9.0-24_linux64_openblas
  libcblas           conda-forge/linux-64::libcblas-3.9.0-24_linux64_openblas
  libcublas          conda-forge/linux-64::libcublas-12.6.3.3-hbd13f7d_1
  libcufft           conda-forge/linux-64::libcufft-11.3.0.4-hbd13f7d_0
  libcurand          conda-forge/linux-64::libcurand-10.3.7.77-hbd13f7d_0
  libcusolver        conda-forge/linux-64::libcusolver-11.7.1.2-hbd13f7d_0
  libcusparse        conda-forge/linux-64::libcusparse-12.5.4.2-hbd13f7d_0
  libexpat           conda-forge/linux-64::libexpat-2.6.3-h5888daf_0
  libffi             conda-forge/linux-64::libffi-3.4.2-h7f98852_5
  libgcc             conda-forge/linux-64::libgcc-14.2.0-h77fa898_1
  libgcc-ng          conda-forge/linux-64::libgcc-ng-14.2.0-h69a702a_1
  libgfortran        conda-forge/linux-64::libgfortran-14.2.0-h69a702a_1
  libgfortran-ng     conda-forge/linux-64::libgfortran-ng-14.2.0-h69a702a_1
  libgfortran5       conda-forge/linux-64::libgfortran5-14.2.0-hd5240d6_1
  libhwloc           conda-forge/linux-64::libhwloc-2.11.1-default_hecaa2ac_1000
  libiconv           conda-forge/linux-64::libiconv-1.17-hd590300_2
  liblapack          conda-forge/linux-64::liblapack-3.9.0-24_linux64_openblas
  libmagma           conda-forge/linux-64::libmagma-2.8.0-h0af6554_0
  libmagma_sparse    conda-forge/linux-64::libmagma_sparse-2.8.0-h0af6554_0
  libmpdec           conda-forge/linux-64::libmpdec-4.0.0-h4bc722e_0
  libnvjitlink       conda-forge/linux-64::libnvjitlink-12.6.77-hbd13f7d_1
  libopenblas        conda-forge/linux-64::libopenblas-0.3.27-pthreads_hac2b453_1
  libprotobuf        conda-forge/linux-64::libprotobuf-5.27.5-h5b01275_2
  libsqlite          conda-forge/linux-64::libsqlite-3.46.1-hadc24fc_0
  libstdcxx          conda-forge/linux-64::libstdcxx-14.2.0-hc0a3c3a_1
  libstdcxx-ng       conda-forge/linux-64::libstdcxx-ng-14.2.0-h4852527_1
  libtorch           conda-forge/linux-64::libtorch-2.4.1-cuda120_h1d34654_302
  libuuid            conda-forge/linux-64::libuuid-2.38.1-h0b41bf4_0
  libuv              conda-forge/linux-64::libuv-1.49.2-hb9d3cd8_0
  libxml2            conda-forge/linux-64::libxml2-2.12.7-he7c6b58_4
  libzlib            conda-forge/linux-64::libzlib-1.3.1-hb9d3cd8_2
  llvm-openmp        conda-forge/linux-64::llvm-openmp-19.1.2-h024ca30_0
  markupsafe         conda-forge/linux-64::markupsafe-3.0.2-py313h8060acc_0
  mkl                conda-forge/linux-64::mkl-2023.2.0-h84fe81f_50496
  mpc                conda-forge/linux-64::mpc-1.3.1-h24ddda3_1
  mpfr               conda-forge/linux-64::mpfr-4.2.1-h90cbb55_3
  mpmath             conda-forge/noarch::mpmath-1.3.0-pyhd8ed1ab_0
  nccl               conda-forge/linux-64::nccl-2.23.4.1-h52f6c39_0
  ncurses            conda-forge/linux-64::ncurses-6.5-he02047a_1
  networkx           conda-forge/noarch::networkx-3.4.1-pyhd8ed1ab_0
  numpy              conda-forge/linux-64::numpy-2.1.2-py313h4bf6692_0
  openssl            conda-forge/linux-64::openssl-3.3.2-hb9d3cd8_0
  pip                conda-forge/noarch::pip-24.2-pyh145f28c_1
  python             conda-forge/linux-64::python-3.13.0-h9ebbce0_100_cp313
  python_abi         conda-forge/linux-64::python_abi-3.13-5_cp313
  pytorch            conda-forge/linux-64::pytorch-2.4.1-cuda120_py313h6ccb88c_302
  readline           conda-forge/linux-64::readline-8.2-h8228510_1
  setuptools         conda-forge/noarch::setuptools-75.1.0-pyhd8ed1ab_0
  sleef              conda-forge/linux-64::sleef-3.7-h1b44611_0
  sympy              conda-forge/noarch::sympy-1.13.3-pyh2585a3b_104
  tbb                conda-forge/linux-64::tbb-2021.13.0-h84d6215_0
  tk                 conda-forge/linux-64::tk-8.6.13-noxft_h4845f30_101
  typing_extensions  conda-forge/noarch::typing_extensions-4.12.2-pyha770c72_0
  tzdata             conda-forge/noarch::tzdata-2024b-hc8b5060_0
  xz                 conda-forge/linux-64::xz-5.2.6-h166bdaf_0
```

Notice that the build information to the right of `pytorch` includes the string `cuda120`:

```
pytorch            conda-forge/linux-64::pytorch-2.4.1-cuda120_py313h6ccb88c_302
```

This means we are downloading the GPU-enabled version of PyTorch. NVIDIA dependencies such as `cudnn` are resolved automatically. Suppose that instead we saw the line:

```
pytorch            conda-forge/linux-64::pytorch-2.4.1-cpu_mkl_py313hbc6f0e9_102
```

This is the CPU-only version of PyTorch built against the [Intel Math Kernel Library (MKL)](https://en.wikipedia.org/wiki/Math_Kernel_Library). If you have used the `CONDA_OVERRIDE_CUDA` environment variable and still see this, check that a CUDA 12 version of the package actually exists in the conda-forge [package repo](https://conda-forge.org/packages/). You can check the package builds in an existing conda environment by running the `conda list` command with the environment activated.

---

**NOTE**

It is recommended to install as many packages as possible in one go when you first create a `conda` environment for a project, as this reduces the chance of unresolvable dependency conflicts down the line.

---

## Testing the conda environment

Batch jobs on ARC4 are scheduled using SGE, which is covered in the [ARC documentation](https://arcdocs.leeds.ac.uk/usage/batchjob.html). Here is a simple job script for testing your GPU-enabled conda environment on a single NVIDIA V100:

```bash
#! /bin/bash -l

# Test GPU-enabled PyTorch

# Job name
#$ -N test_GPU

# Output files
#$ -o output.log
#$ -e error.log

# Run in the current directory (-cwd)
#$ -cwd

# Request some time- min 15 mins - max 48 hours
#$ -l h_rt=00:15:00

# Request GPU node
#$ -l coproc_v100=1

# Get email at start and end of the job
#$ -m be

# Now run the job
unset GOMP_CPU_AFFINITY KMP_AFFINITY
nvidia-smi
conda activate pytorch-gpu
python test_GPU.py
```

There are a few things to notice in this script.

We are using a Bash login shell (hence the `-l` argument in the shebang). This means that our `.bashrc` is sourced at the start of the job and adds `conda` to our path. This is required for standalone conda installations such as miniconda or miniforge. If you are using the anaconda module on ARC4, you will instead need to use the lines:

```bash
module load anaconda
source activate pytorch-gpu
```

Since this is a test script, we are only requesting 15 minutes. You will likely have to wait a long time for this to run as it is, so it's not recommended to increase this unless you know the job will need more time!

The single V100 GPU is requested with `#$ -l coproc_v100=1`. This can be set to a maximum of four cards (the number per GPU node). Unless your code is specifically set up to use multiple GPUs, this should be set to one.

The line `unset GOMP_CPU_AFFINITY KMP_AFFINITY` disables CPU affinity hints from the scheduler, as explained in the [ARC documentation](https://arcdocs.leeds.ac.uk/usage/examples.html?highlight=gomp_cpu#threaded-but-gets-upset-by-openmp-hints). Without it, it's likely you'll see some warnings when using Python.

We call the NVIDIA System Management Interface with `nvidia-smi` to print information about the GPU(s) in use to our job output. This is a simple way to verify that the job has access to the GPU and that the required NVIDIA drivers are working correctly.

Finally, we run our Python test script `test_GPU.py`. You may already have your own PyTorch code that you want to test, however the following code has been verified to run correctly on ARC4:

```python
import torch
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the size of the matrices
matrix_size = 4096

# Create random matrices
A = torch.randn(matrix_size, matrix_size, device=device)
B = torch.randn(matrix_size, matrix_size, device=device)

# Warm-up
for _ in range(10):
    C = torch.matmul(A, B)

# Benchmark
start_time = time.time()
for _ in range(100):
    C = torch.matmul(A, B)
torch.cuda.synchronize()  # Wait for all operations to finish

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time taken for 100 matrix multiplications: {elapsed_time:.4f} seconds")
print(f"Average time per multiplication: {elapsed_time / 100:.4f} seconds")
```

The script runs a simple matrix-multiplication benchmark using PyTorch tensors on the GPU.

---

**NOTE**

Try incrementally increasing the `matrix_size` parameter in the test script. Your jobs will eventually fail with an out-of-memory error.

Memory usage in deep learning can be hard to predict, as it's a function of both the data throughput per batch and of the complexity of the neural network itself. Often, trial and error is needed to find a performant configuration within the limits of the hardware.

---

## Frequently asked questions

