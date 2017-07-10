import csv


source_path = r'C:\Users\innopolis\Desktop\IntershipBCI\OpenBCI\OpenBCI_Python-master\data\Alpha waves\raw'
dest_path = r'C:\Users\innopolis\Desktop\IntershipBCI\OpenBCI\OpenBCI_Python-master\data\Alpha waves'

def process_openbci_raw_files(source_path,dest_path):
	from os.path import isfile, join
	from os import listdir

	onlyfiles = [f for f in listdir(source_path) if isfile(join(source_path, f))]
	for name in onlyfiles:
		with open(source_path+r'\\'+name) as x:
			f = open(dest_path+r"\processed_"+name,"w")

			for row in x:
				if row[0] is '%':
					pass
				else:
					row2 = eval('['+row[0:-14]+']')
					f.write(str(row2[1])+","+str(row2[2])+"\n")

