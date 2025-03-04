try:
    broj=float(input("Unesite broj izmedu 0.0 i 1.0: "))
    if(broj<0.0 or broj>1.0):
        print("Unjeli ste broj izvan zadanih granica! ")
    else:
        if(broj<1.0 and broj>=0.9):
            print("A")
        elif(broj>=0.8):
            print("B")
        elif(broj>=0.7):
            print("C")
        elif(broj>=0.6):
            print("D")
        else:
            print("F")
except ValueError:
    print("Unesite broj!!!")