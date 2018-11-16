import subprocess
bashCommand = "python generate_tfrecord.py --csv_input=data/train.csv  --output_path=data/train.record"
output = subprocess.check_output(['bash','-c', bashCommand])
print(output)

bashCommand = "python generate_tfrecord.py --csv_input=data/test.csv  --output_path=data/test.record"
output = subprocess.check_output(['bash','-c', bashCommand])
print(output)
