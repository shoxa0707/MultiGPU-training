# MultiDevice and MultiGPU training with PyTorch

As a data scientist or software engineer you may be faced with the challenge of processing large datasets or building complex models that require significant computational power One solution to this challenge is to use multiple GPUs to parallelize the workload and speed up the training process In this article we will explore how to use multiple GPUs in PyTorch a popular deep learning framework.

### Why Use Multiple GPUs?

Training deep learning models can be computationally intensive, requiring significant amounts of time and resources. Using multiple GPUs can help speed up the training process by parallelizing the workload. By distributing the workload across multiple GPUs, each GPU can process a smaller portion of the data, reducing the overall training time.

## Usage

Training MultiGPU in one Device:

```bash
python multigpu.py epoch save_every
```

Training MultiGPU in MultiDevice:

```console
  Usage: torchrun --nnodes=3 --nproc_per_node=2 --node_rank=n --master_addr=ip_address --master_port=1224 multinode2.py epoch save_every [options]...

    --nnodes                     Number of devices you want to use.
    --nproc_per_node             Number of GPUs of each device.
    --node_rank                  Rank of device of you run command. This is from 0 to nnodes.
    --master_addr                Master device ip address.
    --master_port                Master device's any free port. Default: 29400
```

In this part, all devices must be one network. And run above command in all devices.

## Installation

- Pytorch with GPU

#### Requirements

- Python >= 3.8
- Linux (Ubuntu 20.04)
- Nvidia GPU
