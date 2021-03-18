# pip install boto3
import boto3
import re
client = boto3.client('batch')

response = client.register_job_definition(
    jobDefinitionName='cpuheater-gym-microrts',
    type='container',
    containerProperties={
        'image': 'cpuheater/gym-microrts:latest',
        'vcpus': 1,
        'memory': 2000,
    },
    retryStrategy={
        'attempts': 3
    },
    timeout={
        'attemptDurationSeconds': 1800
    }
)
print(response)
print("job definition created \n=============================")

