import csv

with open(r'C:\Users\innopolis\Desktop\Intership BCI\OpenBCI\OpenBCI_Python-master\data\Vitaly_EEG_FP_2.txt') as x:
	f = open("processed_Vitaly_EEG_FP_2.txt","w")
	for row in x:
		row2 = eval('['+row[0:-14]+']')
		f.write(str(row2[1])+","+str(row2[2])+"\n")
