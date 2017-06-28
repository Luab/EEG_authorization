import csv

with open(r'C:\Users\innopolis\Desktop\Intership BCI\OpenBCI\OpenBCI_Python-master\data\Bulat_EEG_FP.txt') as x:
	f = open("processed_Bulat_EEG_FPBulat_EEG_FP.txt","w")
	for row in x:
		row2 = eval('['+row[0:-14]+']')
		f.write(str(row2[1])+","+str(row2[2])+","+str(row2[3])+"\n")
