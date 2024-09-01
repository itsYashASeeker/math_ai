from pix2text import Pix2Text
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re

images=['./math.png','./math-2.png','./math-3.png','./math-4.png']



p2t = Pix2Text.from_config()

equations=[]

for image in images:
    equations.append(p2t.recognize_formula(image))

def correct_ocr_errors(eq):
    eq = re.sub(r'(\d)\s+(\d)', r'\1\2', eq)
    # eq = eq.replace('l', '1').replace('O', '0')
    return eq

def process_equation(eq_str):
    eq_str = correct_ocr_errors(eq_str)
    
    eq_str = eq_str.replace("^{", "**(").replace("}", ")")
    eq_str = eq_str.replace(r"\pi", "pi").replace(r"\times", "*").replace("^", "**")
    
    transformations = (standard_transformations + (implicit_multiplication_application,))
    
    if '=' in eq_str:
        lhs, rhs = eq_str.split('=', 1) 
        try:
            lhs_parsed = parse_expr(lhs, transformations=transformations, evaluate=False)
            rhs_parsed = parse_expr(rhs, transformations=transformations, evaluate=False)
            equation = sp.Eq(lhs_parsed, rhs_parsed)
            return equation
        except Exception as e:
            raise ValueError(f"Error parsing equation parts: {e}")
    else:
        try:
            expr = parse_expr(eq_str, transformations=transformations, evaluate=False)
            return expr
        except Exception as e:
            raise ValueError(f"Error parsing expression: {e}")

def solve_or_evaluate(eq):
    if isinstance(eq, sp.Equality):
        symbols = eq.free_symbols
        if len(symbols) == 0:
            return "True" if eq.lhs == eq.rhs else "False"
        elif len(symbols) == 1:
            symbol = symbols.pop()
            solution = sp.solve(eq, symbol)
            return solution
        else:
            solution = sp.solve(eq)
            return solution
    else:
        try:
            evaluated = eq.evalf()
            return evaluated
        except Exception as e:
            return f"Cannot evaluate expression: {e}"

for eq_str in equations:
    try:
        processed_eq = process_equation(eq_str)
        print(f"\nOriginal Equation: {eq_str}")
        print(f"Processed Equation: {processed_eq}")
        result = solve_or_evaluate(processed_eq)
        print(f"Result: {result}")
    except Exception as e:
        print(f"\nError processing equation '{eq_str}': {e}")