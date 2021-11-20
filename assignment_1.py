# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 10:38:57 2021

@author: wakef
"""
import numpy as np
a = np.array([2,3,4])
print("\nArray A:\n",a)
print("\nArray A Data Type:\n", a.dtype)

b = np.array([1.2, 3.5, 5.1])
print("\nArray B Data Type:\n", b.dtype)

# Code for Exercise 1
e1 = np.array([1,2,3,4])
print("\nExercise 1\n Array e1:\n", e1)
print("\nArray e1 Data Type:\n", e1.dtype)

c = np.zeros((2,3))
d = np.ones((3,4))
e = np.empty((9))

print("\nArray C:\n",c)
print("\nArray C Data Type:\n", c.dtype)
print("\nArray D:\n",a)
print("\nArray D Data Type:\n", d.dtype)
print("\nArrayE:\n",a)
print("\nArrayEA Data Type:\n", e.dtype)

#Code for Exercise 2
b = np.zeros((2,7))
print("Exercise 2")
print("\nArray B:\n",b)

f = np.arange(10, 30, 5 )
g = np.linspace( 0,2,9 )

print("\nArray F:\n",f)
print("\nArray F Data Type:\n", f.dtype)
print("\nArray G:\n",g)
print("\nArray G Data Type:\n", g.dtype)

#Code for Exercise 3
c = np.arange( 1, 23, 2.5)

print("Exercise 3")
print("\nArray C:\n",c)
print("\nArray C Data Type:\n", c.dtype)

#Code for Exercise 4
d = np.arange(1,10,1)
print("Exercise 4")
print("\nArray D:\n",d)