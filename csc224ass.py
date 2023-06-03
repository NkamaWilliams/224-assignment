# -*- coding: utf-8 -*-
"""
Created on Mon May 29 12:14:04 2023

@author: Williams
"""
import ast
import sympy as sym
import numpy as np
import math

def run():
    print("Welcome to rootfinder and ODE solver. What would you like to do?")
    print("1. Find a root of an equation")
    print("2. Solve a first order ODE")
    choice = int(input("Input choice number:"))
    print()
    if choice == 1:
        findRoot()
    elif choice == 2:
        solveODE()
    else:
        print("Sorry. Your choice is not a valid choice.")

def findRoot():
    stop = "no"
    while stop == "no":
        print("Input the function you want to find the root of below. Ensure 'x' is the variable you use")
        print("E.g. x**2 - 4*x")
        s = "lambda x:"
        s += input("Input function:")
        f = eval(compile(ast.parse(s, mode='eval'), filename='', mode='eval'))
        try:
            f(0)
        except:
            raise Exception("Invalid function! Did you use the variable x?")
    
        print("Which method do you want to use? Input the corresponding value")
        print("Bisection: 1")
        print("Newton Raphson: 2")
        print("Secant: 3")
    
        method = int(input("Input method value:"))
        
        if method == 1:
            a = float(input("Input x1:"))
            b = float(input("Input x2:"))
            tol = float(input("Input tolerance:"))
            print(myBisection(f, a, b, tol))
        elif method == 2:
            x = float(input("Input x:"))
            tol = float(input("Input tolerance:"))
            x1 = sym.Symbol('x')
            fprime = differentiate(f)
            print(mynewtonRaphson(f, fprime, x, tol))
        elif method == 3:
            x0 = float(input("Input x0:"))
            x1 = float(input("Input x1:"))
            tol = float(input("Input tolerance:"))
            itr = int(input("Input the number of iterations:"))
            print(mySecant(f, x0, x1, tol, itr))
        else:
            print("That is an invalid selection")
        
        stop = input("Do you want to solve for another root? Yes or no:").lower()
    return

def solveODE():
    stop = "no"
    while stop == "no":
        print("Using variables x and y input the first order ODE you want to solve:")
        s = "lambda x, y:"
        s += input("Input function:")
        f = eval(compile(ast.parse(s, mode='eval'), filename='', mode='eval'))
    
        try:
            f(0, 0)
        except:
            raise Exception("Did you use only variables x and y?")
    
        print("What method do you want to use to solve it?")
        print("1. Euler Method")
        print("2. Picard Method")
        print("3. Taylor Method")
    
        method = int(input("Input method number:"))
        if method == 1:
            x = float(input("Initial value of x:"))
            y = float(input("Initial value of y:"))
            dx = float(input("Step size (e.g. 0.2):"))
            itr = int(input("Number of iterations:"))
            print(round(euler(x, y, f, dx, itr), 3))
        elif method == 2:
            x = float(input("Initial value of x:"))
            y = float(input("Initial value of y:"))
            x0 = float(input("Particular value of x:"))
            itr = int(input("Number of iterations:"))
            print(round(picard(f, x, y, x0, itr), 3))
        elif method == 3:
            x = float(input("Initial value of x:"))
            y = float(input("Initial value of y:"))
            x0 = float(input("Particular value of x:"))
            dx = float(input("Step size (e.g. 0.2):"))
            order = int(input("Order of Taylor series?"))
            print(round(taylor(f, x, y, x0, order, dx), 3))
        else:
            print("Invalid selection")
        
        stop = input("Do you want to solve for another ODE?Yes or no:").lower()
    return
        
    
def myBisection(f, a, b, tol):    
    if np.sign(f(a)) == np.sign(f(b)):
        print(a, b)
        raise Exception("The chosen scalars", a, "and", b, "cannot bound to a root")
    
    m = (a + b) / 2
    
    if np.abs(f(m)) < tol:
        return m
    
    elif np.sign(f(a)) == np.sign(f(m)):
        return myBisection(f, m, b, tol)
    
    elif np.sign(f(b)) == np.sign(f(m)):
        return myBisection(f, a, m, tol)

def differentiate(f):    
    x = sym.Symbol('x')
    f1 = sym.sympify(f(x))
    fprime = f1.diff(x)
    fprime = sym.lambdify([x], fprime)
    return fprime

    
def mynewtonRaphson(f, fprime, x, tol):
    if np.abs(f(x)) < tol:
        return x
    
    x = x - f(x)/fprime(x)
    return mynewtonRaphson(f, fprime, x, tol)

def mySecant(f, x0, x1, tol, itr):
    err = np.abs(x1 - x0)
    x2 = 0
    
    if err > tol:
        for i in range(itr):
            try:
                x2 = x1 - f(x1) * (x0 - x1) / (f(x0) - f(x1))
            except ZeroDivisionError:
                print("Error, cannot divide by zero")
                print(f'x0: {x0}\nx1: {x1}\nf(x0): {f(x0)}\nf(x1): {f(x1)}')
                break
            x0 = x1
            x1 = x2
            err = np.abs(x1 - x0)
            if err < tol:
                break
    
    return x1


def euler(x, y, f, dx, itr):
    for i in range(itr):
        try:
            print('x = %.4f | y = %.4f | dy/dx = %.4f' %(x, y, f(x, y)))
            y = y + (dx * f(x, y))
            x += dx
        except:
            print("Unknown error occured")
            break
    return round(y, 4)

def picard(d, x0, y0, x, itr):
    x_sym = sym.Symbol('x')
    y_sym = sym.Symbol('y')
    
    f = sym.sympify(d(x_sym, y_sym))
    y = y0
    
    for i in range(itr):
        integral = sym.integrate(f.subs({y_sym:y}), (x_sym, x0, x_sym))
        y = y0 + integral
        
    return y.subs({x_sym:x})

def taylor(d, x0, y0, x, order = 4, step = .2):
    x_sym = sym.Symbol('x')
    y_sym = sym.Symbol('y')
    
    f = sym.sympify(d(x_sym, y_sym))
    
    while x0 < x: 
        f0 = f
        y = y0
        e = str(y)
        e1 = str(f0)
        for j in range(order):
            expr = (step**(j+1)) * f0.subs({x_sym:x0, y_sym:y0}) / (math.factorial((j+1)))
            e += " '+' " + str(expr)
            y += expr
            f0 = f0.diff(x_sym) + f0.diff(y_sym) * f.subs({x_sym:x0, y_sym:y0})
            
            e1 += " + " + str(f0)
        y0 = y
        x0 += step
        print(e1)
        print("=",e)
        print(f'y({x0}) = {round(y,4)}')
        print('\n\n\n')
    return y

if __name__ == "__main__":
    stop = 0
    while stop == 0:
        run()
        print("Would you like to end the program? Or start again?")
        stop = int(input("Enter 1 to stop, 0 to continue"))
        
    