# Real-Time Recurrent Learning with eLSTM

```bash
cd ~/work/rtrl-elstm/reinforcement_learning
uv run python -m torchbeast_carracing.train \
  --savedir saved_models_carracing \
  --xpid rtrl_elstm_env1_seed2 \
  --seed 2 \
  --num_envs 1 \
  --total_steps 1_000_000
```
