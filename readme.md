# BAIT "Learning to Learn"

## Installation
Dev environment is best set up via VS Code remote server (SSH & DevContainer Extensions)

### Building the VM (Docker Host)
1. Get access to https://apu.cloudlab.zhaw.ch
2. Create a Key Pair via Project/Compute/Key Pairs "Create Key Pair"
    - Name: My-SSH-KeyPair
    - Key Type: SSH
3. Create SSH Security Group via Project/Network/Security Group "Create Security Group"
    - Name: SSH
    -> Create Security Group
    Add Rule via "Add Rule":
        - Description: SSH
        - Directions: Ingress
        - Open Port: Port
        - Port: 22
        - Remote: CIDR
        - CIDR: 0.0.0.0/0
4. Launch Instance Project/Compute/Instances "Launch Instance"
    - Instance Name: any
    - Source: 2022-01_Ubuntu_Focal_nVidia-Cuda_Docker
    - Flavour: g1.xlarge.t4
    - Network: Allocate "internal"
    - Security Group: SSH
    - Key Pair: My-SSH-KeyPair
    -> Launch Instance
5. Add FloatingIp Project/Compute/Instances
    - On Instance: Action dropdown (far right) -> Allocate Floating IP
    - choose any

### Dev Environment (VS Code DevContainer)
VS Code Remote - SSH & Remote - Containers Extensions necessary
Connect to Docker Host
1. Add private key (*.pem file) to VS Code
    - copy .pem file to ~/.ssh
2. Add SSH Host
    - ctrl-p -> Remote-SSH: Add new SSH Host
    - Add Floating IP from Docker Host
3. Connect
4. Pull git repo onto Docker host

Launch DevContainer
1. ctrl+p -> Remote Containers: Reopen in Container

