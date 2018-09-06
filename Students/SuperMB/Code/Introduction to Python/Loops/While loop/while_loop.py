square = 1

while square <= 10:
    print(square)    # This code is executed 10 times
    square += 1      # This code is executed 10 times

print("Finished")  # This code is executed once

square = 0
number = 1

while square < 100:
    square = number ** 2
    if(square < 100):
        print(square)
    number += 1
