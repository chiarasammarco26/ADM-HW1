# Say "Hello, World!" With Python

if __name__ == '__main__':
    print("Hello, World!")
    
# Python If-Else

import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())
    if n in range(1,101):
        if (n % 2) == 0:
            if 2 <= n <= 5:
                print ("Not Weird");
            if 6 <= n <= 20:
                print ("Weird");
            if n > 20:
                print ("Not Weird");
        else:
            print ("Weird");
    else:
        raise ValueError

# Arithmetic Operators

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    if 1 <= a <= 10**10:
        if 1 <= b <= 10**10:
            x = a + b
            y = a - b
            z = a*b
            print(x)
            print(y)
            print(z)
        else:
            raise ValueError
    else:
        raise ValueError

# Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    x = a//b
    y = a/b
    print(x)
    print(y)

# Loops

if __name__ == '__main__':
    n = int(input())
    if n in range(1,21):
        for i in range(0,n):
            x = i**2;
            print(x)
    else:
        raise ValueError

# Write a function

def is_leap(year):
    if year >= 1900 and year <= 10**5:
        if (year % 4) == 0:
            if (year % 100) == 0:
                if (year % 400) == 0:
                    return True
                else:
                    return False
            else:
                return True
        else:
            return False

# Print Function

if __name__ == '__main__':
    n = int(input())
    if n in range(1,151):
        lista1 = ""
        for i in range(1,n+1):
            i = str(i)
            lista1 = lista1 + i
        print(lista1)

# List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
    lista1 = []
    for i in range(0,x+1):
        for j in range(0,y+1):
            for k in range(0,z+1):
                if (i+j+k) != n:
                    lista1.append(i)
                    lista1.append(j)
                    lista1.append(k)
    
    listafinale = [lista1[x:x+3] for x in range(0,len(lista1),3)]
    print(listafinale)

# Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    
    arr = sorted(arr)
    nuovoarr = []
    if n in range(2,11):
        for i in range(len(arr)):
            if arr[i] in range(-100,101) and arr[i] < max(arr):
                nuovoarr.append(arr[i])
                x = max(nuovoarr)
    
    print(x)
    
# Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()

grades = student_marks[query_name]
x = sum(grades)/len(grades)

if n in range(2,11) and len(grades)==3:
    for i in grades:
        if i in range(0,101):
            average = "%0.2f"%x

print(average)

# Lists

if __name__ == '__main__':
    N = int(input())
    lista1 = []
    comandi1 = 0
    for i in range(0,N):
        comando = list(map(str, input().split()))
        if "insert" in comando[0]:
            comandi1 = comandi1 + 1
            lista1.insert(int(comando[1]), int(comando[2]))
        elif "remove" in comando[0]:
            comandi1 = comandi1 + 1
            lista1.remove(int(comando[1]))
        elif "append" in comando[0]:
            comandi1 = comandi1 + 1
            lista1.append(int(comando[1]))
        elif comando[0] == "sort":
            comandi1 = comandi1 + 1
            lista1.sort()
        elif comando[0] == "pop":
            comandi1 = comandi1 + 1
            lista1.pop()
        elif comando[0] == "reverse":
            comandi1 = comandi1 + 1
            lista1.reverse()
        else:
            comandi1 = comandi1 + 1
            print(lista1)

# Tuples

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t = tuple(integer_list)
    print(hash(t))
    
# Nested Lists

if __name__ == '__main__':
    studentlist = []
    scores = []
    names = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        names.append(name)
        scores.append(score)
        studentlist.append([name, score])
    sortingscores = sorted(list(scores))
    studentlist.sort()
    
    seclowscore = sortingscores[1]

    if len(names) in range(2,6):
        for [i,j] in studentlist:
            if j == seclowscore:
                print(i)

# sWAP cASE

def swap_case(s):
    news = ""
    if len(s) in range(0,1001):
        for i in s:
            if i.isupper()==True:
                news += i.lower()
            elif i.islower()==True:
                news += i.upper()
            else:
                news += i
    return news

# String Split and Join

def split_and_join(line):
    line = line.split(" ")
    line = "-".join(line)
    return line

# What's Your Name?

def print_full_name(first, last):
    if len(first) <= 10 and len(last) <= 10:
        print("Hello", first, last + "! You just delved into python.")
        
# Mutations

def mutate_string(string, position, character):
    a = list(string)
    a[position] = character
    string = "".join(a)
    return string

# Find a string

def count_substring(string, sub_string):
    count = 0
    if len(string) in range(1,201) and string.isascii()==True:
        for i in range(0, (len(string)-len(sub_string)+1)):
            if string[i:i+len(sub_string)] == sub_string:
                count = count + 1
            else:
                count = count
        return count

# String Validators

if __name__ == '__main__':
    s = input()
    if len(s) in range(1,1000):
        if any(i.isalnum() for i in s):
            print(True)
        else:
            print(False)
        if any(i.isalpha() for i in s):
            print(True)
        else:
            print(False)
        if any(i.isdigit() for i in s):
            print(True)
        else:
            print(False)
        if any(i.islower() for i in s):
            print(True)
        else:
            print(False)
        if any(i.isupper() for i in s):
            print(True)
        else:
            print(False)

# Text Alignment

thickness = int(input())
c = 'H'
if thickness in range(1,50):
    for i in range(thickness):
        print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

    for i in range(thickness+1):
        print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

    for i in range((thickness+1)//2):
        print((c*thickness*5).center(thickness*6))    

    for i in range(thickness+1):
        print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

    for i in range(thickness):
        print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# Text Wrap

def wrap(string, max_width):
    if len(string) in range(1,1000) and max_width in range(1,len(string)):
        wrapper = textwrap.TextWrapper(width=max_width)
        news = textwrap.dedent(text=string)
        final = wrapper.fill(text=news)
    return final

# Designer Door Mat

N, M = map(int, input().split(" "))
if N in range(6,101) and M in range(16,303) and M==3*N and (N%2)!=0:
    line = [(".|."*(2*i+1)).center(M, "-") for i in range(0,N//2)]
    centro = ["WELCOME".center(M, "-")]
    print("\n".join(line + centro + line[::-1]))

# String Formatting

def print_formatted(number):
    space = len(bin(number)[2:])
    if number in range(0,100):
        for v in range(1,n+1):
            dec = str(v).rjust(space," ")
            octal = oct(v)[2:].rjust(space," ")
            hexadecimal = (hex(v)[2:].upper()).rjust(space," ")
            binary = bin(v)[2:].rjust(space," ")
            print(dec, end=" ")
            print(octal, end=" ")
            print(hexadecimal, end=" ")
            print(binary, end=" ")
            print(" ")

# Capitalize!

def solve(s):
    elem = ""
    lista = s.split()
    for i in lista:
        if i[0].isalnum()==True:
            l = i[0].capitalize()
            elem = l + i[1:]
            s = s.replace(i, elem)
    return(s)

# Introduction to Sets

def average(array):
    l = set(array)
    somma = sum(l)
    length = len(l)
    avg = somma/length
    return(round(avg, 3))

# No Idea!

n, m = input().split()
arr = list(map(int, input().split()))
a = set(list(map(int, input().split())))
b = set(list(map(int, input().split())))

happiness = 0

for i in arr:
    if i in a:
        happiness += 1
    elif i in b:
        happiness -= 1
    else:
        happiness == happiness
print(happiness)

# Symmetric Difference

M = int(input())
m = set(list(map(int, input().split(" "))))
N = int(input())
n = set(list(map(int, input().split(" "))))

diff1 = m.difference(n)
diff2 = n.difference(m)
unione = sorted(diff1.union(diff2))

for i in unione:
    print(i)
    
# Set .add()

N = int(input())
s = set()
for i in range(0,N):
    s.add(str(input()))
print(len(s))

# Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
N = int(input())

for i in range(0,N):
    lines = list(map(str, input().split()))
    if "remove" in lines:
        s.remove(int(lines[1]))
    elif "discard" in lines:
        s.discard(int(lines[1]))
    else:
        s.pop()
print(sum(s))

# Set .union() Operation

n = int(input())
rolleng = set(list(map(int, input().split())))
b = int(input())
rollfr = set(list(map(int, input().split())))

print(len(rolleng.union(rollfr)))

# Set .intersection() Operation

n = int(input())
rolleng = set(list(map(int, input().split())))
b = int(input())
rollfr = set(list(map(int, input().split())))

print(len(rolleng.intersection(rollfr)))

# Set .difference() Operation

n = int(input())
rolleng = set(list(map(int, input().split())))
b = int(input())
rollfr = set(list(map(int, input().split())))

print(len(rolleng.difference(rollfr)))

# Set .symmetric_difference() Operation

n = int(input())
rolleng = set(list(map(int, input().split())))
b = int(input())
rollfr = set(list(map(int, input().split())))

print(len(rolleng.symmetric_difference(rollfr)))

# The Captain's Room

K = int(input())
rooms = list(map(int, input().split()))

for i in rooms:
    if rooms.count(i) == 1:
        print(i)
        
# Check Subset

T = int(input())
for t in range(T):
    n = int(input())
    A = set(map(int, input().split()))
    m = int(input())
    B = set(map(int, input().split()))
    print(A.issubset(B))
    
# Check Strict Superset

A = set(map(int, input().split()))
n = int(input())
for i in range(n):
    B = set(map(int, input().split()))
print(A.issuperset(B))

# Set Mutations

a = int(input())
A = set(map(int, input().split()))
N = int(input())

for i in range(N):
    if len(A) in range(1,1000) and N in range(1,100):
        comand = input().split()[0]
        set1 = set(map(int, input().split()))
        if comand=="intersection_update":
            A.intersection_update(set1)
        elif comand=="update":
            A.update(set1)
        elif comand=="difference_update":
            A.difference_update(set1)
        else:
            A.symmetric_difference_update(set1)
print(sum(A))

# collections.Counter()

from collections import Counter

X = int(input())
sizes = list(map(int, input().split()))
conto = Counter(sizes)
N = int(input())
raghumoney = 0
for i in range(N):
    [shoe, prices] = list(map(int, input().split()))
    if shoe in sizes:
        raghumoney = raghumoney + prices
        sizes.remove(shoe)
    else:
        raghumoney += 0
print(raghumoney)

# DefaultDict Tutorial

from collections import defaultdict
d = defaultdict(list)
n,m = map(int, input().split())
for i in range(1,n+1):
    d[input()].append(i)
for j in range(1,m+1):
    ch = str(input())
    if len(d[ch]) > 0:
        print(" ".join(str(x) for x in d[ch]))
    else:
        print(-1)
        
# Collections.namedtuple()

from collections import namedtuple

N = int(input())
categorie = input().split()
sommavoti = 0
s = namedtuple("s", categorie)

for i in range(1,N+1):
    ID, MARKS, NAME, CLASS = input().split()
    s1 = s(ID, MARKS, NAME, CLASS)
    sommavoti = sommavoti + int(s1.MARKS)
avg = sommavoti/N
print(avg)

# Collections.OrderedDict()

from collections import OrderedDict

itms = OrderedDict()
N = int(input())

for i in range(N):
    item = input().split()
    net_price = int(item[-1])
    item_name = " ".join(item[:-1])
    if itms.get(item_name):
        itms[item_name] += net_price
    else:
        itms[item_name] = net_price

for i in itms.keys():
    print(i, itms[i])
    
# Collections.deque()

from collections import deque

N = int(input())
d = deque()
l = ""
for i in range(N):
    s = list(input().split())
    if s[0]=="append":
        d.append(s[1])
    elif s[0]=="appendleft":
        d.appendleft(s[1])
    elif s[0]=="pop":
        d.pop()
    elif s[0]=="popleft":
        d.popleft()
print(" ".join(d))

# Calendar Module

import calendar
calendar.Calendar(0)
date = input().split()
year = int(date[2])
month = int(date[0])
day = int(date[1])
wday = calendar.weekday(year, month, day)

if wday == 0:
    print("MONDAY")
elif wday == 1:
    print("TUESDAY")
elif wday == 2:
    print("WEDNESDAY")
elif wday == 3:
    print("THURSDAY")
elif wday == 4:
    print("FRIDAY")
elif wday == 5:
    print("SATURDAY")
else:
    print("SUNDAY")
    
# Exceptions

T = int(input())
for i in range(T):
    try:
        a,b = map(int, input().split())
        print(a//b)
    except Exception as exc:
        print("Error Code:", exc)

# Zipped!

N, X = map(int, input().split())
marks = []
for i in range(X):
    mark = list(map(float, input().split()))
    marks.append(mark)
x = zip(*marks)
for j in x:
    print(sum(j)/len(j))
    
# Athlete Sort

import math
import os
import random
import re
import sys

if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    
    s = sorted(arr, key=lambda row:row[k])
    for j in s:
        print(" ".join(str(i) for i in j))
        
# ginortS

S = str(input())
s = sorted([x for x in S])

lower = []
upper = []
odd = []
even = []

for i in s:
    if i.isalpha()==True:
        if i.islower()==True:
            lower.append(i)
        elif i.isupper()==True:
            upper.append(i)
    else:
        if (int(i)%2)==0:
            even.append(i)
        else:
            odd.append(i)

lower = sorted(lower)
upper = sorted(upper)
even = sorted(even)
odd = sorted(odd)

new = lower+upper+odd+even
print("".join(x for x in new))

# Map and Lambda Function

cube = lambda x:x**3

def fibonacci(n):
    fib = [0,1,1]
    for i in range(3,n):
        fib.append(fib[i-1]+fib[i-2])
    return(fib[0:n])

# Detect Floating Point Number

import re
T = int(input())
for i in range(T):
    print(bool(re.match(r'^[-+]?[0-9]*\.[0-9]+$', str(input()))))
    
# Re.split()

regex_pattern = r"[,.]+"

# Group(), Groups() & Groupdict(): I checked the Hackerrank discussions for this exercise, I didn't know how to use the given code.

import re
S = str(input())
p = re.compile(r"([\dA-Za-z])(?=\1)")
s = p.search(S)

if s:
    print(s.group(1))
else:
    print(-1)
    
# Re.findall() & Re.finditer(): this was partly taken from Discussions.

import re
S = str(input())

x = re.finditer(r"(?<=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])([AEIOUaeiou]{2,})(?=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])", S)
m = [i for i in map(lambda x: x.group(), x)]
if m != []:
    print(*m, sep="\n")
else:
    print(-1)

# Validating Roman Numerals

regex_pattern = r"(M{0,3})(C[DM]|D?C{0,3})(X[LC]|L?X{0,3})(I[VX]|V?I{0,3})$"

# Validating phone numbers

import re
N = int(input())
for i in range(N):
    s = str(input())
    if re.match(r"7|8|9",s) and s.isdigit() and len(s)==10:
        print("YES")
    else:
        print("NO")
        
# Re.start() & Re.end()

import re
s = str(input())
k = str(input())
if re.search(r""+k+"", s):
    for m in re.finditer(r"(?=("+k+"))", s):
        print(f"({m.start(1)}, {m.end(1)-1})")
else:
    print("(-1, -1)")

# Hex Color Code

import re

pattern = r'[^^](#[\da-fA-F]{3}\b|#[\da-fA-F]{6}\b)'
for _ in range(int(input())):
    s = input()
    ms = re.findall(pattern, s)
    if len(ms):
        for m in ms:
            print(m)

# HTML Parser - Part 1

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(f"Start : {tag}")
        for name,value in attrs:
            print(f"-> {name} > {value}")

    def handle_startendtag(self, tag, attrs):
        print(f"Empty : {tag}")
        for name,value in attrs:
            print(f"-> {name} > {value}")

    def handle_endtag(self, tag):
        print(f"End   : {tag}")


parser = MyHTMLParser()
for _ in range(int(input())):
    parser.feed(input())
parser.close()

# HTML Parser - Part 2

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if data != '\n':
            if "\n" in data:
                print(">>> Multi-line Comment")
                print(data)
            else:
                print(">>> Single-line Comment")
                print(data)
                  
    def handle_data(self, data):
        if not data == '\n':
            print(f">>> Data")
            print(data)  
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Detect HTML Tags, Attributes and Attribute Values

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self,tag,attrs):
        print(tag)
        for i in attrs:
            print('->',i[0],'>',i[1])

 
parser=MyHTMLParser()
for x in range(int(input())):
    parser.feed(input())

# Validating UID

T = int(input())
t = [str(input()) for i in range(T)]
for x in t:
    upper = []
    digits = []
    repeat = []
    for j in x:
        if j.isalnum()==True:
            if j.isupper()==True:
                upper.append(j)
            elif j.isdigit()==True:
                digits.append(j)
        for z in x:
            if j==z:
                repeat.append(j)
if len(x)==10 and len(upper) >= 2 and len(upper) >= 3 and len(repeat)==0:
    print("Valid")
else:
    print("Invalid")
    
# XML 1 - Find the Score

def get_attr_number(node):
    a = len(node.attrib)
    return a + sum([get_attr_number(x) for x in node])

# XML2 - Find the Maximum Depth

maxdepth = 0
def depth(elem, level):
    global maxdepth
    if level == maxdepth:
        maxdepth += 1
    for i in elem:
        depth(i, level+1)

# Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        a = []
        for i in sorted(l):
            if len(i)==11:
                i = i[1:]
                a.append(i)
            elif len(i)==12:
                i = i[2:]
                a.append(i)
            elif len(i)==13:
                i = i[3:]
                a.append(i)
            else:
                i = i
                a.append(i)
        a = sorted(a)
        for j in a:
            print("+91", j[0:5], j[5:10])
    return fun

# Decorators 2 - Name Directory

from operator import itemgetter

def person_lister(f):
    def inner(people):
        for i in people:
            i[2] = int(i[2])
        people.sort(key=itemgetter(2))
        return map(f, people)
    return inner

# Arrays

def arrays(arr):
    a = numpy.array(arr, float)
    return a[::-1]

# Shape and Reshape

import numpy
lista = list(map(int, input().split()))
my_array = numpy.array(lista)
print(numpy.reshape(my_array, (3,3)))

# Transpose and Flatten

import numpy

N,M = map(int, input().split())
my_array = numpy.array([input().split() for i in range(N)], int)
print(my_array.transpose())
print(my_array.flatten())

# Concatenate

import numpy

N,M,P = map(int, input().split())
array1 = numpy.array([input().split() for i in range(N)],int)
array2 = numpy.array([input().split() for j in range(M)],int)
print(numpy.concatenate((array1, array2)))

# Zeros and Ones

import numpy

numberinp = tuple(map(int, input().split()))

print(numpy.zeros((numberinp), dtype = numpy.int))
print(numpy.ones((numberinp), dtype = numpy.int))

# Eye and Identity

import numpy
numpy.set_printoptions(legacy="1.13")

N,M = map(int, input().split())

# Array Mathematics

import numpy

N,M = map(int, input().split())
A = numpy.array([input().split() for i in range(N)], int)
B = numpy.array([input().split() for j in range(N)], int)

print(numpy.add(A,B))
print(numpy.subtract(A,B))
print(numpy.multiply(A,B))
print(A//B)
print(numpy.mod(A,B))
print(numpy.power(A,B))
print(numpy.eye(N,M))

# Floor, Ceil and Rint

import numpy
numpy.set_printoptions(legacy="1.13")

A = numpy.array(input().split(), float)
print(numpy.floor(A))
print(numpy.ceil(A))
print(numpy.rint(A))

# Sum and Prod

import numpy

N,M = map(int, input().split())
arr = numpy.array([input().split() for i in range(N)], int)
s = numpy.sum(arr, axis=0)
print(numpy.prod(s))

# Min and Max

import numpy

N,M = map(int, input().split())
arr = numpy.array([input().split() for i in range(N)], int)
minimum = numpy.min(arr, axis=1)
print(numpy.max(minimum))

# Mean, Var, and Std

import numpy

N,M = map(int, input().split())
arr = numpy.array([input().split() for i in range(N)], int)
print(numpy.mean(arr, axis=1))
print(numpy.var(arr, axis=0))
print(round(numpy.std(arr),11))

# Dot and Cross

import numpy

N = int(input())
A = numpy.array([input().split() for i in range(N)], int)
B = numpy.array([input().split() for i in range(N)], int)
print(numpy.dot(A,B))

# Inner and Outer

import numpy

A = numpy.array([input().split()], int)
B = numpy.array([input().split()], int)
print(int(numpy.inner(A,B)))
print(numpy.outer(A,B))

# Polynomials

import numpy

P = list(map(float, input().split()))
x = int(input())
print(numpy.polyval(P, x))

# Linear Algebra

import numpy

N = int(input())
A = numpy.array([input().split() for i in range(N)], float)
print(round(numpy.linalg.det(A),2))