from eegparser import parse_files_without_events

filenames = ["processed_Bulat_ECG_v2,v3.txt","processed_Vitaly_ECG_v2,v3.txt"]
def f():
    return 42
data = parse_files_without_events(filenames, 2,f)
for raw in data:
    print(raw.info['subject_info'])