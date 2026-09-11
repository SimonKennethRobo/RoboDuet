# RoboDuet 云端 Stage 1 使用手册

## 1. 登录与日常启动

```bash
ssh -p 30071 root@183.147.142.40
cd /root/gpufree-data/roboduet-conda

# 进入已配置好的环境；实际以 simon 用户运行
./run.sh
```

进入后已自动激活 `isaacgym`，工作目录是 RoboDuet。用 `exit` 返回 root 的 SSH shell。

```bash
whoami
which python
conda info --envs
```

预期用户为 `simon`，Python 为 `/home/simon/miniconda3/envs/isaacgym/bin/python`。无需使用 Docker 或 Apptainer。

## 2. 正式启动 Stage 1

在 root 的 SSH shell 中执行：

```bash
cd /root/gpufree-data/roboduet-conda
./train.sh stage1_main \
  --train_stage stage1 --dyna_gait \
  --num_envs 4096 --num_learning_iterations 100000
```

这个命令由 tmux 托管，SSH 断开后仍继续运行；默认 headless、W&B offline。脚本打印 session、日志文件及退出状态文件的路径。任务名只能包含字母、数字、下划线或连字符；正在运行的任务名不能重复使用。

需要较短的运行时可改 `--num_learning_iterations`。不要用 `--debug` 替代环境数设置，它还会开启视频等调试行为。默认机器人为当前代码的 `go2_x5`。

前台调试示例：

```bash
./run.sh python -u scripts/auto_train.py \
  --train_stage stage1 --dyna_gait --headless --offline \
  --num_envs 256 --num_learning_iterations 3 --run_name quick_check
```

## 3. 查看状态与停止

```bash
tmux list-sessions
tmux attach -t rd-stage1_main
# 按 Ctrl+B，再按 D：退出查看，训练继续

# 使用 train.sh 打印的实际文件名
tail -f logs/stage1_main-YYYYMMDD-HHMMSS.log
nvidia-smi
```

自然结束或报错后，同名 `.exit` 文件保存退出码：0 表示正常结束。tmux session 在任务结束后消失，日志和 checkpoint 保留。

要停止训练，进入对应 tmux session 后按 Ctrl+C。停止前确认最近 checkpoint 已落盘；中断不会保证额外保存一次。不要用 `kill -9` 作为日常停止方式。

## 4. 路径与配置

| 内容 | 路径 |
|---|---|
| 启动脚本和使用手册 | `/root/gpufree-data/roboduet-conda/` |
| 当前代码 | `/root/gpufree-data/RoboDuet` |
| 对齐的项目路径 | `/home/simon/Projects/WBC/RoboDuet`，链接到上述代码 |
| Conda 根目录 | `/home/simon/miniconda3`，链接到数据盘上的 `roboduet-conda/miniconda3` |
| Conda 环境 | `/home/simon/miniconda3/envs/isaacgym` |
| IsaacGym SDK | `/home/simon/Apps/Simulator/Isaac/isaacgym`，链接到数据盘上的 `roboduet-conda/isaacgym` |
| checkpoint 与训练统计 | `/root/gpufree-data/RoboDuet/runs/` |
| 离线 W&B 记录 | `/root/gpufree-data/RoboDuet/wandb/` |
| 终端日志、退出状态 | `/root/gpufree-data/roboduet-conda/logs/` |
| 每个后台任务的实际命令 | `/root/gpufree-data/roboduet-conda/jobs/` |
| 编译缓存 | `/home/simon/.cache`，链接到数据盘上的 `roboduet-conda/runtime/cache` |
| 原服务器代码，未修改 | `/root/Projects/WBC/RoboDuet` |

代码基准提交：`0f02d9d476fc2e3e08b4b6cc9e60e19259bf0775`，云端分支 `cloud-verified`。origin 指向原服务器代码仓库；本机未推送的新提交不会自动同步。

项目代码不在 Conda 包内。可以直接用 vim 修改代码/配置，再重新启动训练。更换工作副本：

```bash
ROBODUET_REPO=/另一个可被simon访问的/RoboDuet ./run.sh
```

## 5. 环境版本与权限

- Ubuntu 24.04；宿主 CUDA Toolkit 12.8，NVIDIA 驱动 580.126.09。
- RTX 4090，24 GB 显存；服务器分配约 14 核 CPU、50 GiB 内存。
- Python 3.8.20、pip Torch 2.3.1+cu121、NumPy 1.23.5。
- Torch 使用 CUDA 12.1 运行库；它与系统 nvcc 显示的 12.8 是不同组件。
- IsaacGym Preview 4；PyTorch3D 0.7.7 的 transforms 可用，不含其原生扩展。
- 已安装 vim、zsh、tmux、git、rsync、curl、wget、htop、ripgrep、GCC 12 等工具。
- Torch CPU 线程默认 4；可通过 `ROBODUET_CPU_THREADS` 调整。

`run.sh` 在 root 下会自动通过 runuser 切换为 simon；当前 simon 的 UID/GID 为 1001。新代码、环境和缓存归 simon 所有。为访问数据盘路径，已给 simon 配置 `/root` 的目录穿越 ACL；不需要给 simon 设置登录密码。

## 6. checkpoint 和 W&B

启动日志中的 `Logging to ...` 是实际 run 目录。狗策略 checkpoint 包括：

```text
<run>/checkpoints_dog/ac_weights_000000.pt
<run>/checkpoints_dog/ac_weights_last_dog.pt
```

请连同同一 run 下的 `parameters.pkl` 一起备份；`deploy_model/body_latest_dog.jit` 是导出的 TorchScript。正常结束会保存最后一次权重；常规周期保存间隔由代码的 RunnerArgs.save_interval 控制。

默认离线 W&B 无需登录。在线记录需先设置 `WANDB_API_KEY`，然后：

```bash
ROBODUET_WANDB_MODE=online ./train.sh stage1_online \
  --train_stage stage1 --dyna_gait --num_envs 4096
```

API key 不写入环境包或 jobs 脚本。当前代码显式指定 entity `simon00715`，账号需有访问权限，或自行修改外部代码配置。完全禁用 W&B 可加 `--no_wandb`；这种模式的 run 目录可能叫 `dummy-...`，应以启动日志为准。

## 7. 本次实测结果

验证只覆盖 Stage 1，未启动 Stage 2，也未启动长期正式实验。

- 4090 CUDA 随机张量与矩阵乘法通过。
- GPU PhysX 小场景步进 20 次通过。
- 256 环境训练 3 次 PPO 迭代，退出码 0。
- 4096 环境训练 20 次 PPO 迭代，退出码 0，共 1,966,080 环境步。
- 训练日志累计迭代时间约 34.59 秒，不包括首次加载；运行中采样到显存约 8.8 GiB。
- checkpoint 可加载，全部浮点参数有限；相较第 0 次迭代有 17 个状态张量更新。
- 导出的 `body_latest_dog.jit` 可加载。

4096 环境验证产物：

```text
训练终端日志：
/root/gpufree-data/roboduet-conda/logs/conda_stage1_4096-20260911-021439.log

训练结果：
/root/gpufree-data/RoboDuet/runs/dummy-wff0kxbm/

最后 checkpoint：
/root/gpufree-data/RoboDuet/runs/dummy-wff0kxbm/checkpoints_dog/ac_weights_last_dog.pt
```

这些结果证明训练链路和保存功能可用，不代表 20 次迭代后的策略已收敛。

## 8. 环境打包与迁移

可迁移文件位于 `packages/`：

- `isaacgym-env.tar.gz`：Conda 环境。
- `isaacgym-sdk.tar.gz`：独立 SDK。
- `SHA256SUMS`：校验文件。

同时复制 `restore.sh`、`env.sh`、`run.sh`、`train.sh`、`verify.py` 和 `miniconda.sha256`。可额外复制 `inputs/miniconda.sh`，避免再次下载 Miniconda。

在另一台已装 NVIDIA 驱动、Ubuntu 24.04 x86_64 的服务器上，将这些文件放到 `/root/gpufree-data/roboduet-conda/`，单独准备代码，然后执行：

```bash
cd /root/gpufree-data/roboduet-conda
bash restore.sh
./run.sh python /root/gpufree-data/roboduet-conda/verify.py --gpu --physics
```

restore.sh 会安装系统依赖、建立用户和路径、解压环境并运行 conda-unpack。它拒绝覆盖已存在的 isaacgym 环境。包中的 SDK editable 引用依赖约定路径，因此需要同时恢复 SDK。

已配置的服务器无需再运行 restore.sh。今后修改依赖后，可用 package.sh 重新打包；先移走旧包，脚本默认不覆盖。

环境包约 3.0 GiB，SDK 包约 193 MiB。已校验两个包的 SHA-256，并将环境解压到不同目录运行 conda-unpack；使用恢复后的 Python/Torch 完成了 SDK 导入、旋转变换和 4090 CUDA 矩阵乘法测试。恢复测试目录随后清理，主训练环境保持不变。

打包和恢复机制参考 [conda-pack 官方文档](https://conda.github.io/conda-pack/)。

## 9. 磁盘与备份

```bash
df -h /root/gpufree-data
du -sh /root/gpufree-data/RoboDuet/{runs,wandb}
du -sh /root/gpufree-data/roboduet-conda/{packages,runtime,miniconda3}
```

数据盘约 49 GB。持续训练应关注 checkpoint、视频及 W&B 的空间占用，并按云平台的数据保留规则备份。日志、环境包和代码可独立备份；不要把整个构建临时目录当作训练结果。
