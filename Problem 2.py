# Birthday Cake Candles

def birthdayCakeCandles(candles):
    massimo = max(candles)
    count = 0
    for i in candles:
        if i == massimo:
            count += 1
    return count

# Number Line Jumps

def kangaroo(x1, v1, x2, v2):
    if int(v1)<int(v2):
        return 'NO'
    if int(v1)>int(v2):
        for i in range(10000):
            a = int(x1)+(int(v1)*int(i))
            b = int(x2)+(int(v2)*int(i))
            if a == b:
                return 'YES'
        if a>b:
            return 'NO'
    else:
        return "NO"

# Viral Advertising

def viralAdvertising(n):
    totale = 2
    c = 2
    for i in range(2,n+1):
        totale = (3*totale)//2
        c = c+totale
    return c

# Recursive Digit Sum

def superDigit(n, k):
    somma = 0
    if len(n)==1 and k==1:
        return n
    else:
        for i in n:
            somma += int(i)
        somma = somma*k
        k = 1
        return superDigit(str(somma),1)

# Insertion Sort - Part 1

def insertionSort1(n, arr):
    value = arr[-1]
    indice = n-2
    while indice >= 0 and value < arr[indice]:
        arr[indice+1]=arr[indice]
        print(*arr)
        indice = indice - 1
    arr[indice+1]=value
    print(*arr)
    
# Insertion Sort - Part 2

def insertionSort2(n, arr):
    for i in range(1,len(arr)):
        value = arr[i]
        indice = i-1
        while indice >= 0 and value < arr[indice]:
            arr[indice+1]=arr[indice]
            indice = indice - 1
        arr[indice+1]=value
        print(*arr)