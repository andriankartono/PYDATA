from Aufgabe3_1 import convert_to_dt

convert_to_dt ("1", 4 , 5.0 , dtype = int)
convert_to_dt ((1 ,0) , "a", 15.1516 , dtype = str)

try:
    convert_to_dt (5 , "a", dtype =int , debug = False )
except ValueError:
    print("encountered value error")

try:
    convert_to_dt (5 , "a", dtype =int , debug = True )
except ValueError:
    print("encountered value error")