import glob
import subprocess

qjob_paths = glob.glob('cluster_scripts/qjobs/*.sh', recursive=True)

for path in qjob_paths:
	bashCommand = f"qsub -o output.text -e error.txt {path}"
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
	
	print(output, error)
