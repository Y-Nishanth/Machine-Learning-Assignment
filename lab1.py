def first_question(arr):
    count=0 #count variable
    for i in arr:
        for j in arr:
            if i+j == 10: #checking whether sum of 2 no.s from the list is 10
                count += 1 #incrementing count variable

    print("The count value is:",int(count/2))

def second_question():
    n=int(input("Enter the no. of elements in the list: ")) #input for the no. of elements in the list
    inp_list=[0]*n #creating list with all 0's
    print("Enter the elements")
    for i in range(n):
        inp_list[i]=input()
    diff=0 #variable to calculate the difference between 2 no.s
    for i in inp_list:
        for j in inp_list:
            if int(i)-int(j) > diff: #Checking for the highest difference
                diff = int(i)-int(j)
    print("The range of the given list of real numbers is: ",diff)

def third_question():
    MatrixA = []
    size = int(input("Enter the size of the matrix")) #Size of the matrix

    for i in range(size):
        row_list = []
        for j in range(size):
            number = int(input(f"Enter element in row {i+1} column {j+1}")) #Entering elements of the matrix
            row_list.append(number)
        MatrixA.append(row_list)

    power = int(input("Enter the power to which you want the matrix: ")) #Value of m

    if power > 0:
        for _ in range(power - 1):
            MatrixA = matrix_multiply(MatrixA, size) #Calling the matrix multiplication function

        print("The final matrix is:")
        for i in range(size):
            for j in range(size):
                print(MatrixA[i][j], end=" ")
            print("\n")


def matrix_multiply(matA, size):
    matB = [row.copy() for row in matA]
    result = [[0 for _ in range(size)] for _ in range(size)]

    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += matA[i][k] * matB[k][j]

    return result

def fourth_question():
    sentence = input("Enter the input string: ") #input for the sentence
    dict = {} #initialising dictionary
    count=0 #Count variable
    for i in sentence:
        if i not in dict.keys() and i.isalpha(): #If the letter does not exist in the dictionary and is an alphabet then it continues
            for j in sentence:
                if i == j:
                    count +=1
            dict[i] = count #Updating the dictionary with the count variable
            count=0
    value = max(dict.values()) #finding the max count
    for i in dict:
        if dict[i]== value:
            key = i #The highest occured alphabet
    print("The highest recurring alphabet is ",key," and it occurs ",value," times")


#first_question([2,7,4,1,3,6])
#second_question()
#third_question()
#fourth_question()
