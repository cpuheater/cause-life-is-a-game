SUBMIT_AWS=True

python create_jobs.py --exp-script run_exp.sh \
    --job-queue gym-microrts \
    --job-definition cpuheater-gym-microrts \
    --num-seed 2 \
    --num-vcpu 3 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 100.0 \
    --submit-aws $SUBMIT_AWS
