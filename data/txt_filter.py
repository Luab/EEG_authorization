
with open('1.csv') as x:
	f = open("processed_Vitaly_ECG_v2,v3.csv","c")
	for row in x.read():
		f.write(row[2]+","+row[3])
