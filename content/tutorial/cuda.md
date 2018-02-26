Title: Installing CUDA 9.1, cuDNN 7.0.5 and Tensorflow with GCC 7.3.0 on Debian Buster
Date: 26-02-2018
Category: tutorials

Installing CUDA and related frameworks on unsupported systems can be quite the
hassle. In this post I will detail my findings that eventually led to
succesfully installing CUDA 9.1 and cuDNN 7.0.5 on Debian Buster using GCC
7.3.0. I also built Tensorflow 1.5 for CUDA 9.1, which required some small
adjustments that I will also list.

## System information

    $ uname -a
    Linux baas 4.14.0-3-amd64 #1 SMP Debian 4.14.17-1 (2018-02-14) x86_64 GNU/Linux

    $ lspci | grep -i nvidia
    01:00.0 VGA compatible controller: NVIDIA Corporation GK104 [GeForce GTX 760] (rev a1)

    $ gcc --version
    gcc (Debian 7.3.0-5) 7.3.0

## Install NVIDIA driver, CUDA 9.1 and cuDNN 7.0.5
First off, download the latest NVIDIA driver for your graphics card (I
downloaded 390.25). Also download CUDA 9.1 as run-file, and the cuDNN 7.0.5
tarball. I assume that you start out without any NVIDIA packages on your
machine; if you do have any, you need to remove them first (`sudo apt purge
*nvidia*`).

### Boot into text mode
To avoid conflicts with X, we will first boot into text mode to install these
packages. Open `/etc/default/grub` in your favourite editor and substitute
`"quiet"` with `"text"` in the following line:

    GRUB_CMDLINE_LINUX_DEFAULT="quiet"

Also uncomment the following line:

    #GRUB_TERMINAL=console

Save these changes, and run the following command to update:

    $ sudo update-grub

Write down the output of the following command somewhere (we will need it
later):

    $ systemctl get-default

Now run:

    $ sudo systemctl enable multi-user.target --force
    $ sudo systemctl set-default multi-user.target
    $ sudo reboot

This will boot you up into text mode.

### Install the NVIDIA driver
Head to the directory where you downloaded the NVIDIA driver, and run the
following (replacing the filename with the correct one):

    $ chmod +x NVIDIA-installer.run
    $ sudo ./NVIDIA-installer.run

This should complete succesfully after you work through the prompts.

### Install CUDA
First we need to install some libraries:

    $ sudo apt install libxmu-dev libxmu-headers libxmuu-dev libglu1-mesa-dev

Head to the directory where you downloaded the CUDA run-file and run the
following:

    $ export PERL5LIB=.
    $ chmod +x cuda-installer.run
    $ sudo ./cuda-installer.run --override

Answer `no` when asked if you want to install the NVIDIA driver; answer yes for
everything else (samples are optional). The `--override` flag is added to
circumvent any GCC version mismatching which is bound to happen.

### Install cuDNN
I'm assuming you created the default `/usr/local/cuda` symlink in the last
step.  Head to the directory where you downloaded the cuDNN tarball and run the
following:

    $ tar xzf cudnn-tarball.tgz
    $ sudo cp cuda/include/* /usr/local/cuda/include/
    $ sudo cp cuda/lib64/* /usr/local/cuda/lib64/

You should now have succesfully installed CUDA and cuDNN.

### Append to PATH and LD\_LIBRARY\_PATH
Add the following to your `.bashrc`:

    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

### Reboot into GUI
Revert the changes you made in `/etc/default/grub`: change `"text"` back to
`"quiet"` and comment the line:

    GRUB_TERMINAL=console

Now run:

    $ sudo systemctl set-default <DEFAULT>

where you substitute `<DEFAULT>` with the output of `systemctl get-default`
that you wrote down.

## Build Tensorflow 1.5 for CUDA 9.1
Since Tensorflow 1.5 precompiled binaries are only compatible with CUDA 9.0, we
will need to compile Tensorflow ourselves.

### Prepare build requirements
We need to install `bazel`, some Python libraries and one NVIDIA package.

#### Bazel using binary installer
Requirements:

    $ sudo apt install pkg-config zip g++ zlib1g-dev unzip python

[Download](https://github.com/bazelbuild/bazel/releases) the binary installer
`bazel-<version>-installer-linux-x86_64.sh` and run it:

    $ chmod +x bazel-<version>-installer-linux-x86_64.sh
    $ ./bazel-<version>-installer-linux-x86_64.sh --user

This will install Bazel to `$HOME/bin`. Make sure that is added to your `PATH`
variable.

#### Python dependencies
I used a virtual environment for this.

    $ pip install numpy dev pip wheel

Alternatively, you can use Debian's packages:

    $ sudo apt install python3-numpy python3-dev python3-pip python3-wheel

#### Install libcupti
Run the following:

    $ sudo apt install libcupti-dev

### Configure build
Get the Tensorflow source:

    $ git clone git@github.com/tensorflow/tensorflow
    $ cd tensorflow && ./configure

Answer yes when asked if you want to build Tensorflow with CUDA support. For
CUDA version, enter 9.1. For cuDNN version, enter 7. For compute capability,
[enter whatever your graphics card is capable
of](https://developer.nvidia.com/cuda-gpus).

### Edit CUDA header to allow GCC > 6
Open `/usr/local/cuda/include/crt/host_config.h` and comment out the
following line:

    #error -- unsupported GNU version! gcc versions later than 6 are not supported!

This will keep the compilation from throwing an error because of GCC 7.

### Compile!
Almost there! To compile Tensorflow, run from the top level of the Tensorflow
source directory:

    $ bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

Grab a cup of coffee; this takes a while. Once it's done, run:

    $ bazel-bin/tensorflow/tools/pip_package/build_pip_package /path/to/package

to create a Python `.whl` file. You can install this file with `pip` in any
virtual environment as follows:

    $ pip install /path/to/package/tensorflow-1.5.0-cp36-cp36m-linux_x86_64.whl

The filename of the wheel file may differ for you.
