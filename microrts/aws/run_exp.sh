for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_gridnet_lstm.py \
    --gym-id Microrts10-workerRushAI-gridnet-lstm \
    --total-timesteps 40000000 \
    --rnn-hidden-size 256 \
    --num-steps 256 \
    --clip-coef 0.2 \
    --learning-rate 0.00025 \
    --wandb-project-name microrts10 \
    --prod-mode \
    --wandb-entity cpuheater --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done