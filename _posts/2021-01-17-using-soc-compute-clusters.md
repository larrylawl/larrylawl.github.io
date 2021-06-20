---
title: 'Crash Course on using SoC Compute Clusters'
date: 2021-01-17
permalink: /posts/2021/01/using-soc-compute-clusters/
tags:
  - networks
---
Need access to NUS computing resources but not sure how? Here's a quick crash course!

### Logging In
```
# VPN
ssh larrylaw@xgpc2.comp.nus.edu.sg

# Otherwise through sunfire
ssh larrylaw@sunfire.comp.nus.edu.sg
ssh xgpc2
```

1. To skip tunneling, either use [ssh tunneling](https://stackoverflow.com/questions/57780250/does-vscode-remote-support-double-ssh) or [SoC VPN](https://dochub.comp.nus.edu.sg/cf/guides/network/vpn).
2. Compute cluster hardware configuration [here](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/hardware).
3. Use RSA key to skip typing of password. Guide [here](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/)

### Transfering Data
```
# From tembusu cluster to local 
## VPN
scp -r larrylaw@xgpc2.comp.nus.edu.sg:~/NM2/results/exp-e ./

## Otherwise through sunfire
scp -r larrylaw@sunfire.comp.nus.edu.sg:~/net_75 .
scp -r results/rs-obs/net_75/ larrylaw@sunfire.comp.nus
.edu.sg:~/

# From local to tembusu cluster
scp lab1.tar.gz larrylaw@xcne2.comp.nus.edu.sg:~/
```

### Lazy to manually check cluster availability?
This bash script echos the availability of specified nodes.

```bash
#!/usr/bin/bash

echo "Checking all remote! /prays hard"

declare -a arr=("xgpb0" "xgpc0" "xgpc1" "xgpc2" "xgpc3" "xgpc4" "xgpd0" "xgpd1" "xgpd4" "xgpf11" "cgpa1" "cpga2" "cpga3")

rm output.txt
for node in "${arr[@]}"
do
    echo "$node" >> output.txt
    echo yes | ssh -o ConnectTimeout=10 "larrylaw@$node.comp.nus.edu.sg" nvidia-smi | grep "MiB /" >> output.txt
done

echo "Go get em!"

```
### Development
1. `pyenv` for python version and `pyvenv` for virtual environment
2. `tmux` to keep process running after ending ssh session. Help [here](https://askubuntu.com/questions/8653/how-to-keep-processes-running-after-ending-ssh-session).
3. `nvidia-smi` to check GPU usage (before sending jobs)
4. Remote development on VSCode. Help [here](https://code.visualstudio.com/docs/remote/ssh)
5. Run on specific GPU via prepending `CUDA_VISIBLE_DEVICES=2,3 python xxx.py`

![Comic](/images/comic.png)