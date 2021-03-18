for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_lstm.py \
    --gym-id Microrts10-workerRushAI-lstm \
    --total-timesteps 40000000 \
    --wandb-project-name microrts10 \
    --prod-mode \
    --wandb-entity cpuheater --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done