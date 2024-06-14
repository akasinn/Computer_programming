# メインの実行ファイル
import os
import sys
print(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )) )
project_root_path = os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))
sys.path.append(project_root_path)

import my_library.load_input_data as input_loader
import my_library.load_dictionary as dictionary_loader
import my_library.count_statistics as counter
import my_library.predict_polarity as predictor
import my_library.manage_output as output_manager

sentence_arrays = input_loader.load("../data/processed_data_v1.txt")

d1 = dictionary_loader.load("../extlib/dictionary1.txt")
d2 = dictionary_loader.load("../extlib/dictionary2.txt")

result = []
for i in range(len(sentence_arrays)):
  sentence = sentence_arrays[i]
  count_statistics = counter.count(d1,d2,sentence)
  result.append( predictor.predict(count_statistics) )

output_manager.output(sentence_arrays, result)
