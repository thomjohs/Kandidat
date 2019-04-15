import ML_functions as ml
import supp

data = ml.load_folder("ProcessedData")


print(supp.count_gesture_length(data))