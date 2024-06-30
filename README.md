# README for Adversarial Detection using Hardware-Accelerated AMD GPUs (Team AOHW-224)

## Project Overview

This project aims to accelerate the detection of adversarial attacks on image recognition models using AMD GPUs. Adversarial attacks can manipulate images at the pixel level, confusing machine learning models and causing incorrect classifications. By leveraging the computational power of AMD GPUs, we aim to enhance the speed and efficiency of detecting these adversarial manipulations, ensuring real-time safety in applications such as autonomous vehicles.

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Implementation](#implementation)
- [Hardware and Software Requirements](#hardware-and-software-requirements)
- [Dataset](#dataset)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)

## Motivation

Autonomous vehicles rely heavily on image recognition for decision-making, such as detecting obstacles and recognizing traffic signs. Adversarial attacks can manipulate these images, leading to potentially dangerous decisions by the vehicle. This project focuses on developing a robust system to detect such attacks in real-time, enhancing the safety and reliability of autonomous systems.

## Implementation

The project is implemented using the PyTorch library in Python and utilizes the ResNet50 architecture for adversarial attack detection. The detection system is capable of identifying two types of adversarial attacks:

- **FGSM (Fast Gradient Sign Method)**: Perturbs the image to confuse the model.
- **PGD (Projected Gradient Descent)**: Finds perturbations that can deceive multiple models.

### Key Features:

- **Model Architecture**: ResNet50
- **Optimization**: Utilized PyTorch just-in-time compiler and INT8 quantization for performance improvements.
- **Performance**: Achieved a significant speedup on AMD GPUs, allowing real-time detection (>100 FPS).

## Hardware and Software Requirements

### Hardware:

- **AMD Instinct™ MI210 Accelerator**
  - Architecture: AMD CDNA 2
  - Compute Units: 104
  - Stream Processors: 6656
  - Memory: 64GB HBM2e
  - Memory Bandwidth: 1.64 TB/s
  - PCIe 4.0 x16 interfaces
    
### System Configurations:

| System   | CPU                          | GPU                         | OS               | RAM                        | CUDA/ROCm  | Python / PyTorch |
|----------|------------------------------|-----------------------------|------------------|----------------------------|------------|------------------|
| System 1 | AMD 5600G 6 cores Zen3       | NVIDIA RTX 4070 12 GB       | Ubuntu 22.04 LTS | 128 GB DDR4@3200MT/S       | CUDA 12.4  | 3.11.9 / 2.3     |
| System 2 | AMD 7600X 6 Cores Zen4       | NVIDIA RTX 4080 Super 16 GB | Ubuntu 24.04 LTS | 64 GB DDR5@6000MT/S        | CUDA 12.4  | 3.11.9 / 2.3     |
| System 3 | AMD 3600 6 Cores Zen2        | NVIDIA RTX 3060 12 GB       | Windows 11       | 32 GB DDR4@3600MT/s        | CUDA 12.1  | 3.11.9 / 2.3     |
| System 4 | AMD EPYC 7V13 64 Cores       | AMD MI210 64 GB             | Ubuntu 20.04 LTS | 504 GB DDR4@3200MT/s       | ROCm 6.0   | 3.11.9 / 2.3     |

### Software:

- **Operating System**: Ubuntu 20.04 LTS or later, and Windows 11
- **Python**: 3.11.9
- **PyTorch**: 2.3
- **ROCm**: 6.0

## Dataset

The project uses the **German Traffic Sign Recognition Benchmark (GTSRB)**, a public dataset of traffic sign images. The dataset has been enhanced with adversarially attacked images, increasing the total number of images to over 60,000.

## Results

The optimized solution demonstrates a high detection accuracy of over 99% for multiple types of attacks and achieves a 4.45x speedup compared to the initial implementation. The AMD Instinct MI210 showed a 10% performance increase over the NVIDIA 4080 Super GPU.

## Contributors

- Mihai-Alexandru Andrei, Email: alexandru.andrei@student.unitbv.ro
- Robert Ducă, Email: robert.duca@student.unitbv.ro
- Bogdan-Valentin Floricescu, Email: bogdan.floricescu@student.unitbv.ro
- Ștefan-Cristian Jarcău, Email: stefan.jarcau@student.unitbv.ro
- Ionuț-Alexandru Oprea, Email: ionut-alexandru.oprea@student.unitbv.ro

Coordinator: Lecturer Cătălin Ciobanu, Email: catalin.ciobanu@unitbv.ro

Special thanks to: Associate Prof. Lucian Sasu

## License

This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE.txt) file for details.

---
For detailed information to the demonstration video, which is available here: [Demo Video](https://www.youtube.com/watch?v=MU8g0x7-VMI)
