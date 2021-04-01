for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_gridnet_encode_decode.py \
    --gym-id Microrts10-workerRushAI-ppo_gridnet_encode_decode \
    --total-timesteps 40000000 \
    --learning-rate 0.00025 \
    --wandb-project-name microrts10 \
    --prod-mode \
    --wandb-entity cpuheater --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done