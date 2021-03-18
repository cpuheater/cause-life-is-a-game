# pip install boto3
import boto3
import re
client = boto3.client('batch')

print("creating job queue")
response = client.create_job_queue(
     jobQueueName='gym-microrts-cpu',
     state='ENABLED',
     priority=100,
     computeEnvironmentOrder=[
         {
             'order': 100,
             'computeEnvironment': 'rl-cpu'
         }
     ]
 )
print(response)
print("job queue created \n=============================")

# print("creating job definition")

