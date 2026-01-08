# Lerobot-integration with BiARX5

## BiARX5 Robot lerobot-teleoperate command

```python
lerobot-teleoperate \
    --robot.type=bi_arx5 \
    --robot.enable_tactile_sensors=true \
    --teleop.type=mock_teleop \
    --fps=30 \
    --debug_timing=false \
    --display_data=true
```

```python
lerobot-teleoperate \
    --robot.type=arx5_follower \
    --robot.control_mode=cartesian_control \
    --robot.enable_tactile_sensors=false \
    --teleop.type=mock_teleop \
    --fps=30 \
    --debug_timing=false \
    --display_data=true
```

## Flexiv Rizon4 Robot with Flare Gripper teleoperate by Spacemouse command

```bash
lerobot-teleoperate \
    --robot.type=flexiv_rizon4 \
    --robot.flare_gripper_mac_addr="e2b26adbb104" \
    --robot.control_mode=cartesian_motion_force_control \
    --teleop.type=spacemouse \
    --fps=30 \
    --display_data=true \
    --debug_timing=true
```

## Flexiv Rizon4 Robot with Flare Gripper teleoperate by Pico4 command

```bash
lerobot-teleoperate \
    --robot.type=flexiv_rizon4 \
    --robot.flare_gripper_mac_addr="e2b26adbb104" \
    --robot.control_mode=cartesian_motion_force_control \
    --teleop.type=pico4 \
    --fps=30 \
    --display_data=true \
    --dryrun=true
```

## Flexiv Rizon4 Robot with Xense Flare teleoperate by Xense Flare command

```bash
lerobot-teleoperate \
    --robot.type=flexiv_rizon4 \
    --robot.flare_gripper_mac_addr="e2b26adbb104" \
    --robot.control_mode=cartesian_motion_force_control \
    --teleop.type=xense_flare \
    --teleop.mac_addr="6ebbc5f53240" \
    --fps=20 \
    --display_data=true \
    --dryrun=true
```

## xense_flare Robot teleoperate by Mock Teleop command

### 1e892b82baa5 -another mac addr

```python
lerobot-teleoperate \
    --robot.type=xense_flare \
    --robot.mac_addr="6ebbc5f53240" \
    --teleop.type=mock_teleop \
    --fps=20 \
    --display_data=true \
    --dryrun=false
```

## Xense-Flare Robot lerobot-record command

```python
lerobot-record \
    --robot.type=xense_flare \
    --robot.mac_addr=6ebbc5f53240 \
    --dataset.repo_id=Vertax/xense_flare_pick_and_place_cube_20260105 \
    --dataset.num_episodes=50 \
    --dataset.single_task="pick up cubes in rgb order from the table and place them in the blue box" \
    --dataset.fps=20 \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=xense_flare \
    --robot.mac_addr=6ebbc5f53240 \
    --dataset.repo_id=Vertax/xense_flare_open_lock_20260108 \
    --dataset.num_episodes=20 \
    --dataset.single_task="open the lock with the key" \
    --dataset.fps=20 \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=xense_flare \
    --robot.mac_addr=6ebbc5f53240 \
    --dataset.repo_id=Vertax/xense_flare_pick_and_place_cubes_20260104 \
    --dataset.num_episodes=50 \
    --dataset.single_task="pick up cubes in rgb order from the table and place them in the blue box" \
    --dataset.fps=20 \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=xense_flare \
    --robot.mac_addr=6ebbc5f53240 \
    --dataset.repo_id=Vertax/xense_flare_cucumber_peeling \
    --dataset.num_episodes=3 \
    --dataset.single_task="peel a cucumber with a vegetable peeler" \
    --dataset.fps=20 \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=xense_flare \
    --robot.mac_addr=6ebbc5f53240 \
    --dataset.repo_id=Vertax/xense_flare_replay_test \
    --dataset.num_episodes=2 \
    --dataset.single_task="xense flare traj replay test" \
    --dataset.fps=20 \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

## BiARX5 Robot lerobot-record command

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Vertax/xense_bi_arx5_tie_shoelaces \
    --dataset.num_episodes=100 \
    --dataset.single_task="tie shoelaces" \
    --dataset.fps=30 \
    --dataset.episode_time_s=300 \
    --display_data=false \
    --resume=true \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --robot.enable_tactile_sensors=true \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Xense/xense_bi_arx5_tie_shoelaces \
    --dataset.num_episodes=5 \
    --dataset.single_task="tie shoelaces" \
    --dataset.fps=30 \
    --dataset.episode_time_s=300 \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --robot.enable_tactile_sensors=true \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Vertax/xense_bi_arx5_tie_shoelaces_tactile \
    --dataset.num_episodes=100 \
    --dataset.single_task="tie shoelaces"
    --dataset.fps=30 \
    --dataset.episode_time_s=300 \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Vertax/xense_bi_arx5_tie_shoelaces_high_quality \
    --dataset.num_episodes=100 \
    --dataset.single_task="tie shoelaces" \
    --dataset.fps=30 \
    --dataset.episode_time_s=300 \
    --display_data=false \
    --resume=true \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Vertax/xense_bi_arx5_tie_shoelaces_1027 \
    --dataset.num_episodes=100 \
    --dataset.single_task="tie shoelaces" \
    --dataset.fps=30 \
    --dataset.episode_time_s=300 \
    --display_data=false \
    --resume=true \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Vertax/xense_bi_arx5_tie_white_shoelaces_1028 \
    --dataset.num_episodes=100 \
    --dataset.single_task="tie shoelaces" \
    --dataset.fps=30 \
    --dataset.episode_time_s=300 \
    --display_data=false \
    --resume=true \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Vertax/xense_bi_arx5_tie_white_shoelaces_1030_no_adjust \
    --dataset.num_episodes=100 \
    --dataset.single_task="tie shoelaces" \
    --dataset.fps=30 \
    --dataset.episode_time_s=300 \
    --display_data=false \
    --resume=true \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --robot.enable_tactile_sensors=true \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Vertax/lerobot040_test_bi_arx5 \
    --dataset.num_episodes=5 \
    --dataset.single_task="test" \
    --dataset.fps=30 \
    --dataset.episode_time_s=300 \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

--------------------------------

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --robot.enable_tactile_sensors=true \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Vertax/lerobot040_pick_and_place_chip_bi_arx5_1204 \
    --dataset.num_episodes=10 \
    --dataset.single_task="pick up a potato chip and place it into the chips container" \
    --dataset.fps=30 \
    --dataset.episode_time_s=60 \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

## BiARX5 Robot lerobot-replay command

```python
lerobot-replay \
    --robot.type=bi_arx5 \
    --dataset.repo_id=Vertax/lerobot040_test_bi_arx5 \
    --dataset.episode=0
```

## BiARX5 Robot lerobot-annotate-reward command

```python
lerobot-annotate-reward \
    --repo-id Xense/xense_bi_arx5_tie_shoelaces \
    --new-repo-id Vertax/test_annotated \
    --push-to-hub
```

## BiARX5 Robot lerobot-train command act

```python
lerobot-train \
  --dataset.repo_id=Vertax/bi_arx5_pick_and_place_cube \
  --policy.type=act \
  --output_dir=outputs/train/act_bi_arx5_pick_and_place_cube \
  --job_name=act_bi_arx5_pick_and_place_cube \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=Vertax/act_bi_arx5_pick_and_place_cube \
  --batch_size=32 \
  --steps=200000 \
  --policy.push_to_hub=true \
  --wandb.disable_artifact=true 
```

## BiARX5 Robot lerobot-train command diffusion

```python
lerobot-train \
  --dataset.repo_id=Vertax/bi_arx5_pick_and_place_cube \
  --policy.type=diffusion \
  --output_dir=outputs/train/diffusion_bi_arx5_pick_and_place_cube \
  --job_name=diffusion_bi_arx5_pick_and_place_cube \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=Vertax/diffusion_bi_arx5_pick_and_place_cube \
  --batch_size=16 \
  --steps=100000 \
  --policy.push_to_hub=true \
  --wandb.disable_artifact=true
```

**Note on preview_time:**

Adjust `--robot.preview_time` to reduce jittering:
- 0.03-0.05s: Smoother motion, more delay (recommended for stable movements)
- 0.01-0.02s: More responsive, but may cause jittering
- 0.0: No preview (only for teleoperation/recording)

## BiARX5 diffusion policy lerobot-eval command

```python
lerobot-record  \
  --robot.type=bi_arx5 \
  --robot.inference_mode=true \
  --robot.preview_time=0.0 \
  --robot.id=bi_arx5 \
  --dataset.fps=30 \
  --dataset.episode_time_s=600 \
  --display_data=false \
  --dataset.repo_id=Vertax/eval_diffusion_bi_arx5_pick_and_place_cube \
  --dataset.single_task="pick and place cube" \
  --policy.path=outputs/train/diffusion_bi_arx5_pick_and_place_cube/checkpoints/last/pretrained_model
```
