SUBMIT_AWS=True

python create_jobs.py --exp-script run.sh \
    --job-queue gym-microrts \
    --job-definition gym-microrts \
    --num-seed 4 \
    --num-vcpu 2 \
    --num-memory 13000 \
    --num-hours 100.0 \
    --submit-aws $SUBMIT_AWS
