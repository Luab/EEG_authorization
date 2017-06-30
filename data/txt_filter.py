import csv

with open(r'C:\Users\innopolis\Desktop\Intership BCI\OpenBCI\OpenBCI_Python-master\data\Subject3_Alpha_1_2_channels.txt') as x:
	f = open("processed_Subject3_Alpha_1_2_channels.txt","w")
	for row in x:
		row2 = eval('['+row[0:-14]+']')
		f.write(str(row2[1])+","+str(row2[2])+"\n")
